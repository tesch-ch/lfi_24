import os
import time
import torch
# import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm


def train_eval_model(model_name, data_dir,
                     batch_size, epochs, lr, momentum,
                     cuda_force=False,
                     log: list|None=None,
                     wandb_id=None, wandb_project="LfI24",
                     safe_model=False, safe_model_file=""):
    # TODO: time, extern model classes for load with state dict
    # Store parameters for logging
    params = {
    "model_name": model_name,
    "data_dir": data_dir,
    "batch_size": batch_size,
    "epochs": epochs,
    "lr": lr,
    "momentum": momentum,
    "cuda_force": cuda_force,
    "log": log,
    "wandb_id": wandb_id,
    "wandb_project": wandb_project,
    "safe_model": safe_model,
    "safe_model_file": safe_model_file
    }


    # Cuda setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if cuda_force:
            raise ValueError("CUDA is not available.")
          
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset
    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                     transform=transform)
    val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                   transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=4)
    
    # Model setup
    if model_name == "baseline":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Freeze all layers in the network
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")

    # Weights and Biases (wandb) setup
    if wandb_id is not None:
        import wandb
        wandb.init(project=wandb_project,
                   id=wandb_id,
                   resume="must",
                   config={"batch_size": batch_size,
                           "epochs": epochs,
                           "lr": lr,
                           "momentum": momentum})

    # Keeping track of the best model. We go after precision, as class 0 is
    # non_smile and class 1 is smile (alphabetical order, assigned to by
    # pytorch's ImageFolder).
    best_state_dict = None
    best_precision = -1.0
    best_metrics = {}

    # Training loop setup
    pbar_epochs = tqdm(total=epochs, desc="Epochs",
                       position=0, leave=True)
    pbar_train = tqdm(total=len(train_loader), desc="Training",
                      position=1, leave=True)
    pbar_val = tqdm(total=len(val_loader), desc="Validation",
                    position=2, leave=True)
    
    # Training loop
    time_start_training = time.time()
    for epoch in range(epochs):
        time_start_epoch = time.time()
        # Set progress bars
        pbar_epochs.set_description(f"Epoch {epoch+1}/{epochs}")
        pbar_train.reset(total=len(train_loader))
        pbar_val.reset(total=len(val_loader))

        model.train()
        running_loss = 0.0 # training loss
        running_loss_val = 0.0 # validation loss
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar_train.update(1)
        
        # Evaluation on the validation set
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss_val = criterion(outputs, labels)
                running_loss_val += loss_val.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                pbar_val.update(1)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary')
        
        # Logging
        avg_epoch_loss = running_loss / len(train_loader)
        avg_epoch_loss_val = running_loss_val / len(val_loader)
        epoch_log = {"epoch": epoch + 1, 
                     "train_loss": avg_epoch_loss,
                     "val_loss": avg_epoch_loss_val,
                     "accuracy": accuracy, 
                     "precision": precision, 
                     "recall": recall, 
                     "f1_score": f1}
        
        if log is not None:
            log.append(epoch_log)
        if wandb_id is not None:
            wandb.log(epoch_log)

        # Check for best model
        if precision > best_precision:
            best_state_dict = model.state_dict()
            best_precision = precision
            best_metrics = epoch_log
        
        time_duration_epoch = time.time() - time_start_epoch
        pbar_epochs.set_postfix_str(
            f"Epoch {epoch+1}:\n"
            f"Train Loss: {avg_epoch_loss:.2f}, Precision: {precision:.2f}, "
            f"Duration: {time_duration_epoch:.2f}s")
        pbar_epochs.update(1)


    # Save best model
    if safe_model:
        torch.save(best_state_dict, safe_model_file)
        print(f"Best model's state dict saved at {safe_model_file}")

    # Finish run on wandb
    if wandb_id is not None:
        wandb.finish()

    # Return log and metrics


    return {"params": params, "best_metrics": best_metrics, "log": log}
