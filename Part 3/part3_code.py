import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from tqdm.auto import tqdm  
import wandb

# =============================================================================
# Function: get_resnet18_model
# -----------------------------------------------------------------------------
# Loads a pretrained ResNet18 model from torchvision and modifies it to suit 
# CIFAR-100 images:
#  - The original first convolution uses a 7x7 kernel and stride=2, which 
#    is too aggressive for 32x32 images. We change it to a 3x3 kernel with 
#    stride=1 and padding=1.
#  - The initial max pooling layer is removed (replaced with Identity) so 
#    that the spatial dimensions are not reduced too quickly.
#  - The final fully connected layer is replaced so that the model outputs 
#    logits for 100 classes.
# =============================================================================
def get_resnet18_model(num_classes=100, pretrained=False):
    # Load the standard ResNet18 architecture; if pretrained=True, it loads weights
    model = models.resnet18(pretrained=pretrained)
    # Modify the first convolution to be more suitable for 32x32 CIFAR images.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Replace the max pooling layer with Identity to avoid downsampling.
    model.maxpool = nn.Identity()
    # Retrieve the number of input features for the final fully connected layer.
    num_features = model.fc.in_features
    # Replace the final fully connected layer with one that outputs 100 classes.
    model.fc = nn.Linear(num_features, num_classes)
    return model

# =============================================================================
# Function: mixup_data
# -----------------------------------------------------------------------------
# Applies MixUp augmentation to a batch:
#  - Draws a mixing coefficient lambda from a Beta(alpha, alpha) distribution.
#  - Shuffles the batch and creates a convex combination of each input with
#    another random input.
#  - Returns the mixed inputs along with the original labels and their shuffled
#    counterparts, and lambda.
# =============================================================================
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    # Create a random permutation of indices for mixing.
    index = torch.randperm(batch_size).to(x.device)
    # Create the mixed inputs as a convex combination of original and shuffled images.
    mixed_x = lam * x + (1 - lam) * x[index, :]
    # Get the corresponding labels.
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# =============================================================================
# Function: train_one_epoch
# -----------------------------------------------------------------------------
# Trains the model for one epoch using MixUp augmentation:
#  - Puts the model in training mode.
#  - Iterates over the training DataLoader.
#  - For each batch, moves data to device, applies MixUp, performs forward pass,
#    computes loss (weighted sum from mixed labels), and backpropagates.
#  - Updates the model parameters and the OneCycleLR scheduler after each batch.
#  - Tracks and displays running loss and accuracy (using original labels for accuracy).
# =============================================================================
def train_one_epoch(epoch, model, loader, optimizer, criterion, scheduler, device, mixup_alpha=0.4):
    model.train()  
    running_loss = 0.0  
    correct = 0         
    total = 0           
    
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for inputs, labels in progress_bar:
        # Move inputs and labels to GPU/CPU
        inputs, labels = inputs.to(device), labels.to(device)
        # Apply MixUp augmentation on the batch
        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=mixup_alpha)
        optimizer.zero_grad()  # Zero gradients before backpropagation
        outputs = model(inputs)  # Forward pass: compute predictions
        # Compute MixUp loss as a weighted sum of the loss for both sets of labels
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        loss.backward()         # Backward pass: compute gradients
        optimizer.step()        # Update model parameters
        scheduler.step()        # Update learning rate as per OneCycleLR policy

        running_loss += loss.item()  # Accumulate loss
        # Calculate predictions (using the original order for accuracy computation)
        _, predicted = outputs.max(dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        # Update progress bar with average loss and accuracy so far
        progress_bar.set_postfix(loss=running_loss/(len(loader)), acc=100.*correct/total)
    # Compute average loss and accuracy for the epoch
    return running_loss/len(loader), 100.*correct/total

# =============================================================================
# Function: validate
# -----------------------------------------------------------------------------
# Evaluates the model on the validation dataset:
#  - Sets the model to evaluation mode (disabling dropout and gradient computation).
#  - Iterates over the validation DataLoader, computing loss and accuracy.
#  - Returns the average loss and accuracy.
# =============================================================================
def validate(model, loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
     
    with torch.no_grad():
        # Create a progress bar over the validation DataLoader
        progress_bar = tqdm(loader, desc="Validate", leave=False)
        for inputs, labels in progress_bar:
            # Move data to the target device
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            running_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            # Update progress bar with current loss and accuracy
            progress_bar.set_postfix(loss=running_loss/(len(loader)), acc=100.*correct/total)
    # Calculate overall validation loss and accuracy
    return running_loss/len(loader), 100.*correct/total

# =============================================================================
# Function: main
# -----------------------------------------------------------------------------
# The main function sets up configuration, data transforms, datasets, model,
# optimizer, scheduler, and training loop. It uses wandb for experiment logging.
# After training, it evaluates the best model on the test set and performs OOD evaluation.
# =============================================================================
def main():
    # Configuration dictionary with hyperparameters and settings.
    CONFIG = {
        "model": "ResNet18_Pretrained_MixUp_AA_AdamW",  
        "batch_size": 128,           
        "base_lr": 1e-4,             # Base learning rate for AdamW (lower LR typical for fine-tuning)
        "epochs": 50,                
        "num_workers": 4,            # Number of workers for data loading
        "device": "cuda" if torch.cuda.is_available() else "cpu", 
        "data_dir": "./data",        
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge", 
        "seed": 42,                 
        "mixup_alpha": 0.4,          # Alpha parameter for MixUp augmentation
    }

    # Set the random seeds for reproducibility.
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # ---------------------------
    # Data Transformations for Training:
    # - RandomCrop: Crops a random 32x32 patch from the image with 4 pixels of padding.
    # - RandomHorizontalFlip: Randomly flips the image horizontally.
    # - AutoAugment: Applies automatic augmentation using the CIFAR10 policy.
    # - ColorJitter: Randomly changes brightness, contrast, saturation, and hue.
    # - ToTensor: Converts the image to a PyTorch tensor.
    # - Normalize: Normalizes the image tensor using mean and std (set to 0.5 for each channel).
    # ---------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # ---------------------------
    # Data Transformations for Testing/Validation:
    # - Only converts to tensor and normalizes the image.
    # ---------------------------
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # ---------------------------
    # Load CIFAR-100 Training Dataset:
    # Downloads the dataset (if not already present) and applies training transforms.
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
    # Load CIFAR-100 Test Dataset:
    # Downloads the test set and applies testing transforms.
    # ---------------------------
    testset = torchvision.datasets.CIFAR100(
        root=CONFIG["data_dir"], train=False, download=True, transform=transform_test
    )

    # ---------------------------
    # Create DataLoaders for training, validation, and testing.
    # ---------------------------
    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # ---------------------------
    # Instantiate the pretrained ResNet18 model:
    # The model is loaded with pretrained weights (from ImageNet) and modified to output 100 classes.
    # ---------------------------
    model = get_resnet18_model(num_classes=100)
    model = model.to(CONFIG["device"])

    # ---------------------------
    # Define the loss function:
    # CrossEntropyLoss with label smoothing (default smoothing is 0.0, here set to 0.1).
    # ---------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ---------------------------
    # Setup the AdamW optimizer:
    # AdamW optimizer with a base learning rate and weight decay is used.
    # ---------------------------
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["base_lr"], weight_decay=5e-4)

    # ---------------------------
    # Setup the OneCycleLR scheduler:
    # Adjusts the learning rate dynamically over training iterations.
    # Scheduler updates after every batch, using the total number of epochs and steps per epoch.
    # ---------------------------
    scheduler = OneCycleLR(optimizer,
                           max_lr=CONFIG["base_lr"],
                           steps_per_epoch=len(trainloader),
                           epochs=CONFIG["epochs"])

    # ---------------------------
    # Initialize Weights & Biases (wandb) for experiment tracking.
    # This logs metrics like loss and accuracy during training.
    # ---------------------------
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model)

    best_val_acc = 0.0  # Variable to track the best validation accuracy achieved.
    # ---------------------------
    # Training Loop:
    # For each epoch, the model is trained on the training set and evaluated on the validation set.
    # The scheduler is stepped after each epoch.
    # The best model (based on validation accuracy) is saved.
    # ---------------------------
    for epoch in range(CONFIG["epochs"]):
        # Train the model for one epoch using MixUp augmentation.
        train_loss, train_acc = train_one_epoch(epoch, model, trainloader, optimizer, criterion, scheduler, CONFIG["device"], mixup_alpha=CONFIG["mixup_alpha"])
        # Validate the model on the validation set.
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        # Update the scheduler (adjusts learning rate for next epoch).
        # Note: In OneCycleLR, scheduler.step() is typically called after each batch.
        # Here, we call it per epoch since it was also stepped per batch.
        # ---------------------------
        # Log metrics to wandb.
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
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    wandb.finish()

    # ---------------------------
    # Evaluation on Test Set:
    # Evaluate the best saved model on the test dataset.
    # ---------------------------
    import eval_cifar100  
    import eval_ood      

    # Evaluate the model on the CIFAR-100 test set.
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # Evaluate OOD performance and create a submission CSV.
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")

if __name__ == '__main__':
    main()