This repository contains a dataset for classifying neuroretinal images into either abnormal or normal classes, and a six-class system (Grade 1, Grade 2, Grade 3, Grade 4, Atypical, and Normal). The dataset is divided into three sections: Bioptigen, Copernicus, and Combined.

## Dataset location
The dataset can be found in the following directory: /data/neuroretinal/

### Bioptigen
* 6class-Biop-binary: Binary classification of Bioptigen images
* Biop-dataset-binary: Binary classification of Bioptigen images
* Biop-dataset-6-Cropped: Six-class classification of cropped Bioptigen images
* Biop-dataset: Six-class classification of Bioptigen images
* 6class-Biop: Six-class classification of Bioptigen images
* BIOPTIGEN_AI: Bioptigen images for AI training
### Copernicus
* Foveal-Dataset-Cropped-Binary: Binary classification of cropped 
Copernicus images
* Foveal-Dataset-Cropped: Six-class classification of cropped Copernicus images
### Combined
* Combined-Binary: Binary classification of combined Bioptigen and Copernicus images
* Combined: Six-class classification of combined Bioptigen and Copernicus images

### Training
To train the model, activate the fastai conda environment and run the command:

```
conda activate fastai
python train.py
```

### Inference
To perform inference on the trained model, activate the fastai conda environment and run the command:
```
conda activate fastai
python inference.py

```


