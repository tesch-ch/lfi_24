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

class ModelBaseline(nn.Module):
    """Used as baseline for the transfer learning binary classification task
    at hand. The model is a ResNet50 with pytorch's IMAGENET1K_V2 weights.
    The layers are frozen, except for the last fully connected layer, which
    is replaced by a new one with 2 outputs.
    Input: 3x224x224 image
    """
    def __init__(self):
        super(ModelBaseline, self).__init__()
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for param in self.model.parameters():
            param.requires_grad = False
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2) # Not frozen by default
    def forward(self, x):
        return self.model(x)


class ResNet50Based2FC(nn.Module):
    """Same as above with two fully connected layers.
    """
    def __init__(self):
        super(ResNet50Based2FC, self).__init__()
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for param in self.model.parameters():
            param.requires_grad = False
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 2))
    def forward(self, x):
        return self.model(x)


# influenced by https://github.com/pytorch/vision/blob/main/references/classification/train.py
def train_eval_model(model_name: str,
                     data_dir: str,
                     batch_size: int,
                     epochs: int,
                     momentum: float,
                     lr: float,
                     weight_decay: float=0.0,
                     useCosineAnnealingLR: bool=False,
                     warmup_epochs: int=0,
                     warmup_lr_decay: float=0.0,
                     label_smoothing: float=0.0,
                     n_data_workers: int=4,
                     pin_memory: bool=True,
                     cuda_force: bool=False,
                     log: list|None=None,
                     wandb_id: str|None=None,
                     wandb_project: str="LfI24",
                     safe_model: bool=False,
                     safe_model_file: str=""):

    # Store parameters for logging
    params = {
    "model_name": model_name,
    "data_dir": data_dir,
    "batch_size": batch_size,
    "epochs": epochs,
    "momentum": momentum,
    "lr": lr,
    "weight_decay": weight_decay,
    "useCosineAnnealingLR": useCosineAnnealingLR,
    "warmup_epochs": warmup_epochs,
    "warmup_lr_decay": warmup_lr_decay,
    "label_smoothing": label_smoothing,
    "n_data_workers": n_data_workers,
    "pin_memory": pin_memory,
    "cuda_force": cuda_force,
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
                              num_workers=n_data_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=n_data_workers, pin_memory=pin_memory)
    
    # Model setup
    if model_name == "baseline":
        model = ModelBaseline()
    elif model_name == "resnet50_2fc":
        model = ResNet50Based2FC()
    else:
        raise NotImplementedError(f"Model {model_name} not implemented.")
    
    model = model.to(device)

    # label_smoothing = 0.0 means no smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer Setup
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    
    # Learning rate scheduler for main training
    if useCosineAnnealingLR:
        scheduler_main_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs-warmup_epochs, eta_min=0.0)
    else:
        # Constant scheduler to mimic the behavior of not using a scheduler
        scheduler_main_lr = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=epochs-warmup_epochs)

    # Learning rate scheduler for warmup
    if warmup_epochs > 0:
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_lr_decay, total_iters=warmup_epochs)
        # In order to not manually keep track of warmup and main training,
        # I'll use SequentialILR. It will take care of scheduler switching
        # automatically.
        scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[scheduler_warmup, scheduler_main_lr],
                milestones=[warmup_epochs])
    else:
        scheduler = scheduler_main_lr


    # Weights and Biases (wandb) setup
    if wandb_id is not None:
        import wandb
        # This is simply one log message to the wandb dashboard, nothing is
        # happening about the actual training.
        wandb.init(project=wandb_project,
                   id=wandb_id,
                   resume="must",
                   config={"model_name": model_name,
                           "data_dir": data_dir,
                           "batch_size": batch_size,
                           "lr": lr,
                           "epochs": epochs,
                           "momentum": momentum,
                           "weight_decay": weight_decay,
                           "useCosineAnnealingLR": useCosineAnnealingLR,
                           "warmup_epochs": warmup_epochs,
                           "warmup_lr_decay": warmup_lr_decay,
                           "label_smoothing": label_smoothing,
                           "n_data_workers": n_data_workers,
                           "pin_memory": pin_memory,
                           "cuda_force": cuda_force,
                           "saved_model": safe_model_file})

    # Keeping track of the best model. We go after smile identification which
    # corresponds to precision. Class 0 is # non_smile and class 1 is smile
    # (alphabetical order, assigned to by pytorch's ImageFolder).
    best_state_dict = None
    best_precision = -1.0
    best_metrics = {}

    # Training loop progress bar setup
    pbar_epochs = tqdm(total=epochs, desc="Epochs",
                       position=0, leave=True)
    pbar_train = tqdm(total=len(train_loader), desc="Training",
                      position=1, leave=True)
    pbar_val = tqdm(total=len(val_loader), desc="Validation",
                    position=2, leave=True)
    time_start_training = time.time()
    timestamp_start = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    
    # ********************* Training loop *********************
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

        scheduler.step()
        
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
        time_duration_epoch = time.time() - time_start_epoch
        avg_epoch_loss = running_loss / len(train_loader)
        avg_epoch_loss_val = running_loss_val / len(val_loader)
        epoch_log = {"epoch": epoch + 1,
                     "duration_s": time_duration_epoch,
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
        
        # Update progress bar
        pbar_epochs.set_postfix_str(
            f"Epoch {epoch+1}:\n"
            f"Train Loss: {avg_epoch_loss:.2f}, Precision: {precision:.2f}, "
            f"Duration: {time_duration_epoch:.2f}s")
        pbar_epochs.update(1)
        # ********************* End of training loop *********************

    time_duration_training = time.time() - time_start_training
    # Save best model
    if safe_model:
        torch.save(best_state_dict, safe_model_file)
        print(f"Best model's state dict saved at {safe_model_file}")

    # Finish run on wandb
    if wandb_id is not None:
        wandb.finish()

    return {"params": params,
            "timestamp_start_utc": timestamp_start,
            "duration_min": time_duration_training/60,
            "best_metrics": best_metrics, "log": log}
