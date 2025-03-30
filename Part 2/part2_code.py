
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  
import wandb

# =============================================================================
# Function: get_resnet18_model
# - Loads a ResNet18 model.
# - Modifies the first convolutional layer to better suit CIFAR-100 images (32x32):
#   • Changes kernel size from 7 to 3.
#   • Changes stride from 2 to 1.
#   • Uses padding=1 to maintain spatial dimensions.
# - Replaces the aggressive max pooling layer with Identity (i.e. no pooling).
# - Replaces the final fully connected layer to output 100 classes.
# =============================================================================
def get_resnet18_model(num_classes=100, pretrained=False):
    # Load the standard ResNet18 architecture (optionally with pretrained weights)
    model = models.resnet18(pretrained=pretrained)
    # Modify the first convolutional layer to better handle 32x32 input images.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove the initial max pooling layer to prevent excessive downsampling.
    model.maxpool = nn.Identity()
    # Get the number of input features for the final fully connected layer.
    num_features = model.fc.in_features
    # Replace the final fully connected layer so that the model outputs 100 classes.
    model.fc = nn.Linear(num_features, num_classes)
    return model

# =============================================================================
# Function: train
# - Runs one epoch of training.
# - Iterates over the training DataLoader, processes each batch.
# - Moves inputs and labels to the device (GPU/CPU).
# - Performs a forward pass, computes the loss, and performs backpropagation.
# - Updates the model parameters using the optimizer.
# - Tracks and prints the running loss and training accuracy.
# =============================================================================
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]   
    model.train()               
    running_loss = 0.0          
    correct = 0                 
    total = 0                   
    
    # Wrap the training loader with tqdm for a progress bar display.
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
    for i, (inputs, labels) in enumerate(progress_bar):
        # Move inputs and labels to the target device.
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero out gradients for the current batch.
        optimizer.zero_grad()
        # Perform a forward pass through the model.
        outputs = model(inputs)
        # Compute the loss between model outputs and true labels.
        loss = criterion(outputs, labels)
        # Backpropagate the loss.
        loss.backward()
        # Update the model parameters using the optimizer.
        optimizer.step()
        # Accumulate the loss.
        running_loss += loss.item()
        # Get predicted classes by taking the index with maximum output.
        _, predicted = outputs.max(dim=1)
        # Update the total number of samples processed.
        total += labels.size(0)
        # Update the count of correct predictions.
        correct += predicted.eq(labels).sum().item()
        # Update progress bar with current average loss and accuracy.
        progress_bar.set_postfix({
            "loss": running_loss / (i + 1),
            "acc": 100. * correct / total
        })
    # Compute average loss and accuracy for the epoch.
    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# =============================================================================
# Function: validate
# - Evaluates the model on the validation dataset.
# - Runs the model in evaluation mode (disables dropout, gradient computations, etc.).
# - Iterates over the validation DataLoader, computes loss and accuracy.
# - Returns average loss and accuracy over the validation set.
# =============================================================================
def validate(model, valloader, criterion, device):
    model.eval()                
    running_loss = 0.0          
    correct = 0                 
    total = 0                   
    
    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
        for i, (inputs, labels) in enumerate(progress_bar):
            # Move inputs and labels to the target device.
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass: compute outputs.
            outputs = model(inputs)
            # Compute loss.
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            # Get predicted classes.
            _, predicted = outputs.max(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            # Update progress bar with current loss and accuracy.
            progress_bar.set_postfix({
                "loss": running_loss / (i+1),
                "acc": 100. * correct / total
            })
    # Compute average loss and accuracy.
    val_loss = running_loss / len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

# =============================================================================
# Function: main
# - Main entry point for training and evaluating the model.
# - Sets up configuration, data transformations, datasets, and DataLoaders.
# - Instantiates the model, loss function, optimizer, and learning rate scheduler.
# - Runs the training loop for a fixed number of epochs.
# - Logs metrics using wandb.
# - Saves the best model based on validation accuracy.
# - After training, evaluates the model on the test dataset and performs OOD evaluation.
# =============================================================================
def main():
    # Configuration dictionary containing hyperparameters and settings.
    CONFIG = {
        "model": "ResNet18",                      
        "batch_size": 128,                        
        "learning_rate": 0.1,                     # Initial learning rate for SGD
        "epochs": 50,                             
        "num_workers": 4,                         
        "device": "cuda" if torch.cuda.is_available() else "cpu",  
        "data_dir": "./data",                     # Directory to download and store dataset
        "ood_dir": "./data/ood-test",             # Directory for out-of-distribution test data
        "wandb_project": "sp25-ds542-challenge",    
        "seed": 42,                               
    }

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # ---------------------------
    # Define the data transformations for training.
    # - RandomCrop: Randomly crops images to 32x32 with 4 pixels of padding.
    # - RandomHorizontalFlip: Randomly flips images horizontally.
    # - ToTensor: Converts PIL images to PyTorch tensors.
    # - Normalize: Normalizes tensor values to have mean 0.5 and std 0.5 per channel.
    # ---------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # ---------------------------
    # Define the data transformations for testing/validation.
    # Only conversion to tensor and normalization is applied.
    # ---------------------------
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # ---------------------------
    # Load the CIFAR-100 training dataset using the training transformations.
    # ---------------------------
    trainset_full = torchvision.datasets.CIFAR100(
        root=CONFIG["data_dir"], train=True, download=True, transform=transform_train
    )

    # ---------------------------
    # Split the full training dataset into training (80%) and validation (20%) subsets.
    # ---------------------------
    train_size = int(0.8 * len(trainset_full))
    val_size = len(trainset_full) - train_size
    trainset, valset = random_split(trainset_full, [train_size, val_size])

    # ---------------------------
    # Load the CIFAR-100 test dataset using the test transformations.
    # ---------------------------
    testset = torchvision.datasets.CIFAR100(
        root=CONFIG["data_dir"], train=False, download=True, transform=transform_test
    )

    # ---------------------------
    # Create DataLoaders for training, validation, and testing.
    # DataLoaders handle batching and shuffling.
    # ---------------------------
    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # ---------------------------
    # Instantiate the ResNet18 model.
    # Here, the model is trained from scratch (pretrained=False).
    # The model is modified to suit CIFAR-100 images by adjusting the first conv layer and removing maxpool.
    # The final fully connected layer is replaced to output 100 classes.
    # ---------------------------
    model = get_resnet18_model(num_classes=100, pretrained=False)
    model = model.to(CONFIG["device"])

    # ---------------------------
    # Define the loss function.
    # CrossEntropyLoss is standard for classification tasks.
    # ---------------------------
    criterion = nn.CrossEntropyLoss()

    # ---------------------------
    # Define the optimizer.
    # SGD with momentum (0.9) and weight decay (5e-4) is used.
    # ---------------------------
    optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=0.9, weight_decay=5e-4)

    # ---------------------------
    # Define a learning rate scheduler.
    # CosineAnnealingLR gradually reduces the learning rate over the specified T_max (here, total epochs).
    # ---------------------------
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model)

    best_val_acc = 0.0  # Initialize the best validation accuracy to zero.

    # ---------------------------
    # Training loop:
    # For each epoch, the training function is called to update model parameters.
    # Then, the model is evaluated on the validation set.
    # The learning rate scheduler is updated after each epoch.
    # The best model (highest validation accuracy) is saved.
    # ---------------------------
    for epoch in range(CONFIG["epochs"]):
        # Train the model for one epoch.
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        # Validate the model on the validation dataset.
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        # Update the learning rate.
        scheduler.step()

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        # If the current validation accuracy is the best so far, save the model.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")

        # Print a summary of the epoch.
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    wandb.finish()

    # ---------------------------
    # Evaluation:
    # Evaluate the trained model on the test set and print the test accuracy.
    # Also evaluate out-of-distribution (OOD) performance and create a CSV submission.
    # ---------------------------
    import eval_cifar100  
    import eval_ood       

    # Evaluate on CIFAR-100 test set.
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # Evaluate on OOD data and create a submission CSV.
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")

if __name__ == '__main__':
    main()