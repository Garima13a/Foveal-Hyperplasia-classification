The dataset is divided into three sections: Bioptigen, Copernicus, and Combined.

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

### Gradcam
```
python gradcam.py
```
Publications:

*  H. Kuht, S. Wang, G. Nishad, S. George, G. Maconachie, V. Sheth, Z. Tu, M. Hisaund, R.
McLean, S. Teli,et al. Using artificial intelligence (AI) to classify retinal developmental disor-
ders. Investigative Ophthalmology & Visual Science, 61(7):4030–4030, 2020. [Link](https://iovs.arvojournals.org/article.aspx?articleid=2769372)
*  H. J. Kuht, G. Nishad, S. S. Wang, G. Maconachie, V. Sheth, Z. Tu, M. Hisaund, R. J. McLean,
R. Purohit, S. Teli, et al. A machine learning solution to predict foveal development and visual
prognosis in retinal developmental disorders. Investigative Ophthalmology & Visual Science,
62(8):2739–2739, 2021. [Link](https://iovs.arvojournals.org/article.aspx?articleid=2775782)

