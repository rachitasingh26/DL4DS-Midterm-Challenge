import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  
import wandb

# =============================================================================
# Define a simple Convolutional Neural Network (CNN)
# =============================================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # ---------------------------
        # Convolutional Block 1:
        # - First convolution: from 3 input channels (RGB) to 32 channels,
        #   with a 3x3 kernel and padding of 1 (to keep the spatial dimensions).
        # - Second convolution: from 32 to 64 channels.
        # - MaxPool: halves the spatial dimensions.
        # ---------------------------
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------------------------
        # Convolutional Block 2:
        # - Third convolution: from 64 to 128 channels.
        # - Fourth convolution: from 128 to 256 channels.
        # - Another MaxPool: halves the spatial dimensions again.
        # ---------------------------
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---------------------------
        # Fully Connected Layers:
        # After two pooling operations, the original 32x32 image is reduced to 8x8.
        # With 256 channels, the total number of features is 256 * 8 * 8 = 16384.
        # - fc1: Reduces 16384 features to 512.
        # - fc2: Outputs 100 scores for the 100 classes in CIFAR-100.
        # ---------------------------
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 100)  # Output layer for 100 classes

    def forward(self, x):
        # ---------------------------
        # Pass input through the first convolutional block:
        # Convolution -> ReLU -> Convolution -> ReLU -> Pooling
        # ---------------------------
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        # ---------------------------
        # Pass the result through the second convolutional block:
        # Convolution -> ReLU -> Convolution -> ReLU -> Pooling
        # ---------------------------
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        # ---------------------------
        # Flatten the output tensor for the fully connected layers.
        # The size is reshaped from (batch_size, 256, 8, 8) to (batch_size, 16384).
        # ---------------------------
        x = x.view(x.size(0), -1)  # Alternatively, torch.flatten(x, 1) works too.

        # ---------------------------
        # Pass through fully connected layers:
        # Apply ReLU after fc1, then fc2 to get the final class scores.
        # ---------------------------
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =============================================================================
# Training Function: Runs one epoch on the training set.
# =============================================================================
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]           # Get the device (GPU or CPU)
    model.train()                       # Set model to training mode
    running_loss = 0.0                  # Accumulate loss
    correct = 0                         # Count correct predictions
    total = 0                           # Total number of samples

    # Wrap the training DataLoader with tqdm for progress display.
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    for i, (inputs, labels) in enumerate(progress_bar):
        # Move inputs and labels to the specified device.
        inputs, labels = inputs.to(device), labels.to(device)

        # ---------------------------
        # Zero the gradients from the previous iteration.
        # ---------------------------
        optimizer.zero_grad()

        # ---------------------------
        # Forward pass: compute outputs.
        # ---------------------------
        outputs = model(inputs)

        # ---------------------------
        # Compute the loss between the outputs and the true labels.
        # ---------------------------
        loss = criterion(outputs, labels)

        # ---------------------------
        # Backward pass: compute gradients.
        # ---------------------------
        loss.backward()

        # ---------------------------
        # Update the model parameters.
        # ---------------------------
        optimizer.step()

        # ---------------------------
        # Update statistics for reporting.
        # ---------------------------
        running_loss += loss.item()
        _, predicted = outputs.max(dim=1)   # Get predicted classes
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar with current loss and accuracy.
        progress_bar.set_postfix({
            "loss": running_loss / (i + 1),
            "acc": 100. * correct / total
        })

    # Compute average loss and accuracy over the entire epoch.
    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# =============================================================================
# Validation Function: Evaluates the model on the validation set.
# =============================================================================
def validate(model, valloader, criterion, device):
    model.eval()                        
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass only.
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({
                "loss": running_loss / (i+1),
                "acc": 100. * correct / total
            })

    val_loss = running_loss / len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

# =============================================================================
# Main Function: Sets up configuration, data, model, training loop, and evaluation.
# =============================================================================
def main():
    # Configuration dictionary containing hyperparameters and settings.
    CONFIG = {
        "model": "SimpleCNN",             
        "batch_size": 64,                 
        "learning_rate": 0.01,            
        "epochs": 50,                     
        "num_workers": 4,                 # Number of worker threads for data loading
        "device": "cuda" if torch.cuda.is_available() else "cpu",  
        "data_dir": "./data",             # Directory where data will be stored/downloaded
        "ood_dir": "./data/ood-test",     # Directory for out-of-distribution test data
        "wandb_project": "sp25-ds542-challenge",  
        "seed": 42,                       
    }

    # ---------------------------
    # Define data transformations for training:
    # - RandomCrop with padding to allow random cropping.
    # - RandomHorizontalFlip for data augmentation.
    # - Conversion to tensor.
    # - Normalization with mean and standard deviation for each channel.
    # ---------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),  # Mean for each channel (R, G, B)
                             (0.5, 0.5, 0.5)),# Standard deviation for each channel
    ])

    # ---------------------------
    # Define transformation for test/validation data:
    # - Convert to tensor.
    # - Normalize with the same mean and std as training data.
    # ---------------------------
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])

    # ---------------------------
    # Load the CIFAR-100 training dataset with training transforms.
    # ---------------------------
    trainset_full = torchvision.datasets.CIFAR100(
        root=CONFIG["data_dir"], train=True, download=True, transform=transform_train
    )

    # ---------------------------
    # Split the training dataset into a training set (80%) and a validation set (20%).
    # ---------------------------
    train_size = int(0.8 * len(trainset_full))
    val_size = len(trainset_full) - train_size
    trainset, valset = torch.utils.data.random_split(trainset_full, [train_size, val_size])

    # ---------------------------
    # Load the CIFAR-100 test dataset with test transforms.
    # ---------------------------
    testset = torchvision.datasets.CIFAR100(
        root=CONFIG["data_dir"], train=False, download=True, transform=transform_test
    )

    # ---------------------------
    # Create DataLoaders for training, validation, and test sets.
    # DataLoaders handle batching, shuffling, and parallel data loading.
    # ---------------------------
    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # ---------------------------
    # Instantiate the SimpleCNN model and move it to the target device.
    # ---------------------------
    model = SimpleCNN()
    model = model.to(CONFIG["device"])

    # ---------------------------
    # Define the loss function (CrossEntropyLoss) used for classification.
    # ---------------------------
    criterion = nn.CrossEntropyLoss()

    # ---------------------------
    # Define the optimizer (SGD with momentum and weight decay).
    # The optimizer updates the model parameters based on gradients.
    # ---------------------------
    optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=0.9, weight_decay=5e-4)

    # ---------------------------
    # Define a learning rate scheduler:
    # StepLR reduces the learning rate by a factor of 0.1 every 10 epochs.
    # This helps the model to converge as training progresses.
    # ---------------------------
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    wandb.init(project="-sp25-ds542-challenge", config=CONFIG)
    wandb.watch(model)

    best_val_acc = 0.0  
    # ---------------------------
    # Training loop: iterate over the number of epochs.
    # ---------------------------
    for epoch in range(CONFIG["epochs"]):
        # Train for one epoch and get training loss and accuracy.
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        # Validate the model on the validation set.
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        # Step the learning rate scheduler.
        scheduler.step()

        # Log metrics to wandb for visualization.
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        # If current validation accuracy is the best seen so far, save the model checkpoint.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")

    wandb.finish()

    # ---------------------------
    # After training, evaluate the final model on the test set.
    # The evaluation functions are assumed to be defined in external modules.
    # ---------------------------
    import eval_cifar100   
    import eval_ood        

    # Evaluate test set performance and print the test accuracy.
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # Evaluate OOD performance and create a submission CSV.
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")

if __name__ == '__main__':
    main()