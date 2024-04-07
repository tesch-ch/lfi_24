# Training
Models are trained from within a notebook running on Google Colab.
- `training_local.ipynb`: Can be run without gpu support, just for ensuring a working training script.
- `training.ipynb`

- gunning for precision as class 1 is smiling and want to identify as much smiles as possible

- Balance dataset...

## Models
- ResNet50 Models (pretrained), https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
  - Two output neurons, cross entropy loss
    - baseline: model frozen, one fc with depth 1
      - with some stuff maybe?
    - fine1: model frozen, one fc with depth 2
    - fine2: model frozen, one fc with depth 2 and layer before fc unfrozen
  - Two output neurons allow for easy label smoothing in cross entropy loss function
- YOLOv8 
- No extensive HPO for saving resources...
- Image augmentation
- No extensive cropping, head might get cut...

### Baseline
- batch_size=128
- epochs=100
- opt=sgd
- momentum=0.9
- lr=0.01
- --> second fc makes no difference 

### Optimizations
- learning rate tuning?
- add early stopping
- learning rate scheduler
- in cross entropy loss label smoothing
- weighted loss function

### stuff you could implement
- checkpointing mechanisms



## Finds
- https://github.com/pytorch/vision/tree/main/references/classification
- Pretrained Pytorch Resnet50 Recipe
  - https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
  - https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
  - https://github.com/pytorch/vision/issues/3995
- https://github.com/pytorch/examples/blob/main/imagenet/main.py
- https://imbalanced-learn.org/stable/index.html