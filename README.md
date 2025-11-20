# Trash-Classification

Here is what has happened so far. It is probably going to sound messy so we can arrange a meeting and discuss properly. This is for me to remember and you to get more context.

## Steps:
1. Created Dataset
2. Trained model
3. Saved best accuracy model
4. Created an app.py file for front-end

## Set-up:
The notebook and dataset sit in a ood.explorer (northeastern hpc) directory.</br>
Structure is just:</br>
trash_classification_using_ResNet.ipynb</br>
  |</br>
  Dataset (unzip dataset before uploading. code does not hande that currently. hpc so do not need to worry about it either mostly)</br>
        |</br>
        |- cups</br>
        |- cans</br>
        |- bottles</br>
        |- cardboard</br>

### Step 1:
Dataset currently has 4 directories:
1. cups
2. bottles
3. cardboard
4. cans
The dataset is created by collecting decent images from Kaggle and roboflow datasets.</br>
Each has around or over 1100 images</br>

### Step 2:
Started with training a CNN + MobileNetV2 model but then realized that we have a lot of targets to deal with.</br></br>
Resnet50 is better at training higher complexity tasks with limited dataset. Apparently it is good with distingushing backgrounds and objects (deeper networks) and pattern recognition becasue does not have the vanishing gardient problem (where the model forgets information as it proceeds along the network). Hence ResNet good.</br></br>
Gave us validation (or testing) accuracy of 98% with the current trashclassify.keras file in this repo.</br>

### Step 3:
It needed some work to upload the trashclassify.keras file downloaded from hpc to GitHub.</br>
This was necessary because Streamlit has a very convenient way to update using GitHub and it needs the model file and app.py to run. The model file has to be a zipped keras. Here is how to check that:
```
file trashclassify.keras
```
And the output should be:
```
trashclassify.keras: Zip archive data, at least v2.0 to extract, compression method=store
```
Surprisingly older versions of keras store it in a .h5 format which streamlit does not accept.
So we need keras and tensorflow versions to be:
```
print(tf.__version__)
print(keras.__version__)
2.16.1
3.3.3
```
They are compatible and do the zipping things too.</br>
It took a few tries to upload it to GitHub. Also tried converting to tflite but some model layers were causing issues.</br>
Steps to upload to GitHub using lfs:</br>
1. brew install git-lfs (might have to download from here for windows: https://git-lfs.com/)
2. git lfs install (check installation)
3. git lfs track ".keras"
4. git add .gitattributes (has all extensions that should be treated under lfs)
5. git commit -m "message"
6. git push origin main

If remote is not set:
1. `git remote -v`
2. If blank: `git remote add origin https://github.com/your_username_after_cloning/Trash-Classification.git`
3. `git remote -v` (should show above path as fetch and push)
4. `git push origin main`

If untracked large files already on commit tree:
1. `git clean -n` (shows files)
2. `git clean -f -d` (removes directories and files forecfully)
3. Or,`git filter-repo --strip-blobs-bigger-than 10M --force`

### Step 4:
app.py takes the saved model and uses it on the front-end.</br>
Chose Streamlit because it is meant to be used as front-ends for projects with models in the backend.</br>
https://rise-trash-classification.streamlit.app/

## Optional To-dos:
1. Pipeline:</br>
Think of a pipeline to combine the notebook, dataset and GitHub repo code.</br>
Something to make it more easily reproducible.
2. Dataset:</br>
Add more directories with over 1100 images.</br>
See recycle.jpeg and compost.jpeg to add more directories.
3. Should we add a part to the front-end to keep updating the dataset? Add targets as options to choose from (like wrappers, cups, bottles, etc). When people click on it, they can click pictures of that object and upload it to that directory.</br>
But I don't know how careful people will be about that.
