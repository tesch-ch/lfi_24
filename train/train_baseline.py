import os
import torch
# import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# def train_eval_model(wandb_id, data_dir, batch_size=128, epochs=100, lr=0.01, momentum=0.9):
def train_eval_model(model_name, data_dir,
                     batch_size, epochs, lr, momentum,
                     cuda_force=False,
                     log: list|None=None,
                     wandb_id=None, wandb_project="LfI24",
                     safe_model=False, safe_model_file=""):
    # TODO: log dict, time, best_model, extern model classes for load with state dict

    # Cuda setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if cuda_force:
            raise ValueError("CUDA is not available.")
          
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset
    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                     transform=transform)
    val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                   transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # Model setup
    if model_name == "baseline":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
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

    # Training and Validation
    print("Training started...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Evaluation on the validation set
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary')
        
        # Logging
        avg_epoch_loss = running_loss / len(train_loader)
        epoch_log = {"epoch": epoch + 1, 
                     "loss": avg_epoch_loss, 
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
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss}, Precision: {precision}")


    # Save best model
    if safe_model:
        torch.save(best_state_dict, safe_model_file)
        print(f"Best model's state dict saved at {safe_model_file}")

    # Finish run on wandb
    if wandb_id is not None:
        wandb.finish()

    return {"best_metrics": best_metrics, "log": log}
