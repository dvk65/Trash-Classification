# Trash-Classification using ResNet50
This project develops a multi-class image classification model for identifying common waste items to support proper recycling and reduce contamination in waste streams. The repository includes the dataset uploaded using Git Large File Storage, model experimentation, training pipelines, and a Streamlit-based web interface that performs real-time classification using a fine-tuned ResNet50 model.

## Overview
This repository demonstrates:</br>
1. Data collected from multiple public sources
2. Preprocessing and augmentation for small/medium-sized vision datasets
3. Model discussion (why CNN + MobileNetV2 to ResNet50)
4. Training and evaluation on an HPC cluster
5. Saving and exporting models in Streamlit-compatible format
6. Deployment through a simple, reproducible app.py front-end

## Set-up:
I used our university provided HPC cluster for training the notebook due to python library versions and higher processing resources.</br>
Project Structure:</br>
```
Trash-Classification/
│
├── trash_classification_using_ResNet.ipynb   # Model training notebook
├── trashclassify.keras                       # Saved model (Keras ZIP format)
└── Dataset/                                  # Unzip dataset before re-creating this Structure on HPC
       ├── bottles/
       ├── cans/
       ├── cardboard/
       ├── cups/
       ...
```

### Dataset Creation:
The dataset consists of four waste categories:
1. recycle
2. compost
3. general trash
4. commingled (bottles and cans)
Images were collected from Kaggle and Roboflow public datasets and curated for balanced representation.</br>
Each class contains ~1100+ images, ensuring sufficient coverage for small-scale deep learning experiments.</br>
Future extensions will add categories such as compostable materials. To help us grow the dataset refer the jpeg files in the repo.

### Model Selection and Training:
Initial experiments used a custom CNN + MobileNetV2 (TensorFlow Lite–friendly) model.</br>
These architectures performed well, but with four visually similar classes and complex backgrounds, the models plateaued.</br>
ResNet50 was ultimately selected because:
- Deep residual networks mitigate the vanishing gradient problem
- They capture fine-grained features and background context
- They perform strongly on limited datasets after transfer learning
- They generalize better than lightweight CNNs for multi-class tasks
After fine-tuning, the final model achieved **98% validation accuracy**.</br>
The model is saved as a .keras ZIP-format file compatible with modern TensorFlow/Streamlit deployments which can be checked using:
```
file trashclassify.keras
```
from the same directory where the .keras file is saved.</br>
And the output should be:
```
trashclassify.keras: Zip archive data, at least v2.0 to extract, compression method=store
```
Older versions of keras store it in a .h5 format which streamlit does not accept.</br>
So the required keras and tensorflow versions are:
```
print(tf.__version__)
print(keras.__version__)
2.16.1
3.3.3
```
### Managing Large Model Files (Git LFS):
Streamlit deployments require the model file to be available through app.py.</br>
Because the file exceeds GitHub's standard size limit since it was trained on a HPC, Git LFS (Large File Storage) was used to upload it to Hugging Face at: https://huggingface.co/dvk65/trash-classifier-resnet50.</br>
Steps to upload to Hugging Face using lfs:</br>
1. `brew install git-lfs` (might have to download for windows: https://git-lfs.com/)
2. `git lfs install` (check installation)
3. `git lfs track ".keras"`
4. `git add .gitattributes` (gitattributes has all extensions that should be treated under lfs)
5. `git add trashclassify.keras`
6. `git commit -m "message"`
7. `git push origin main --force`

If large files are already committed, repository cleanup can be performed using:
1. `git clean -n` (shows files)
2. `git clean -f -d` (removes directories and files forecfully)
3. Or,`git filter-repo --strip-blobs-bigger-than 10M --force` (add origin again after using this command)

### Streamlit App Deployment:
Streamlit makes deploying public apps from GitHub fairly easy. To do this, connect your GitHub account to Streamlit (or login to Streamlit with GitHub credentials).</br>
Then go to "Create App" and choose to deploy a public app from GitHub. Select the repo that has your app.py and requirements.txt.</br>
One thing to remember when creating the requirements.txt is to specify the tensorflow version of 2.20.0 since it supports importing models from Hugging Face.</br>
The Streamlit interface (app.py) loads the trained model and provides:
- Camera or file-upload input
- Real-time preprocessing (resizing, normalization)
- Model inference and probability scores
- User-friendly output for recycling decisions
Live Demo: https://rise-trash-classification.streamlit.app/

## Future Work:
1. Reproducible Pipeline:</br>
- Automate dataset download, preprocessing, training, and model saving
- Create workflow to run model and update front-end with a single click or script
2. Dataset Expansion:</br>
- Add more waste directories
- Increase dataset size and diversity
- Introduce synthetic augmentation pipelines
