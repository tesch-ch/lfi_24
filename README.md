# lfi_24

## Select Training Data

## Explore Datasets

## Data Prep
- Binary Classification (two classes: smile and non smile)
- Consolidate the GENKI and UCF files according to the following structure:
  ```
  dataset/
  ├── train/
  │   ├── smile/
  │   │   ├── 1.jpg
  │   │   └── ...
  │   └── non_smile/
  │       ├── 2.jpg
  │       └── ...
  ├── val/
  │   ├── smile/
  │   │   ├── 3.jpg
  │   │   └── ...
  │   └── non_smile/
  │       ├── 4.jpg
  │       └── ...
  └── test/
      ├── smile/
      │   ├── 5.jpg
      │   └── ...
      └── non_smile/
          ├── 6.jpg
          └── ...
  ```
  - Load all the paths to the files in a 


## stuff
- UCF contains some notion of ethnicity (would allow for checking if PoC are included in data...)
- with face object detection find faces in dataset and crop...
- model trained only on genki...
- Data augmentation
  - Standard data augmentation techniques for training on ImageNet include random cropping, horizontal flipping, and color jittering. The images are also resized to 224x224 pixels after augmentation.
  - But, images are goofy anyways, so might not need be super important
  - e.g. (torchvision transforms):
    ```python
    train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.8,1.2), shear=5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize])
    ```
- A big advantage of cosine is that there are no hyper-parameters to optimize, which cuts down our search space. (https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/)
- Choose two output neurons for better comparison between model's probabilities (softmax, cross entropy loss...)
- One vs two output neurons
- RayTune in the future...
- Data might be imbalanced in many ways, such as underrepresented smiling males...

# old readme
## Datasets

### GENKI-4k
- 4000 images
- spanning a wide range of subjects, facial appearance, illumination, geographical, locations, imaging conditions, and camera models
- Smile content (1=smile, 0=non-smile)
- Head Pose (yaw, pitch, and roll parameters, in radians)
- https://inc.ucsd.edu/mplab/398/
- direct data download link (April 24): https://inc.ucsd.edu/mplab/398/media/genki4k.tar
- README:  
    ```
    GENKI 4K
    ------------------------------------------------------------------------------------

    This is the GENKI 4K dataset, collected by the Machine Perception Laboratory,
    University of California, San Diego. This dataset contains 4000 images along 
    with expression (smile=1, non-smile=0) labels and pose labels (yaw, pitch, and roll,
    in radians). The file "labels.txt" contains these labels, and the Nth line of the
    file corresponds to N in the "files" directory.

    We ask that you acknowledge use of this dataset in any papers you publish with the following citation:
    "http://mplab.ucsd.edu, The MPLab GENKI-4K Dataset."
    ```
    - citation: http://mplab.ucsd.edu, The MPLab GENKI-4K Dataset.
    - Nth line of the file corresponds to N in the "files" directory
    - expression (smile=1, non-smile=0) labels and pose labels (yaw, pitch, and roll, in radians)
- labels.txt first 4 rows:  
    ```
    1 -0.021162 0.059530 -0.039662
    1 -0.057745 0.083098 -0.094952
    1 0.095993 0.028798 0.065996
    1 0.000000 0.047124 0.171268
    ```

 ### UCF Selfie Data Set
- 46836 images
- https://www.crcv.ucf.edu/data/Selfie/
- direct download link: https://www.crcv.ucf.edu/data/Selfie/Selfie-dataset.tar.gz
