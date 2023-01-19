from fastai import *
from fastai.vision import *
from fastai.core import *
from gradcam import *
import glob
#This code is for loading pretrained model for 6 class classification 
#This code explains using gradcam for fastai
# https://nbviewer.jupyter.org/github/anhquan0412/animation-classification/blob/master/gradcam-usecase.ipynb




print("let's start")
path1 = "/data/neuroretinal/Combined"
tfms = get_transforms(do_flip=False,flip_vert=True)
print("getting databunch")
data = ImageDataBunch.from_folder(path1, ds_tfms=tfms, size=224)
data.normalize()
print("got databunch")

learn = cnn_learner(data, models.resnet50, metrics=accuracy)
learn.load("/data/neuroretinal/Combined/models/biop-fovnew", strict=False, remove_module=True)
print(data.classes)


Normal=glob.glob('/data/neuroretinal/Combined/train/Normal/*.*')
G1=glob.glob('/data/neuroretinal/Combined/train/Grade 1/*.*')
G2=glob.glob('/data/neuroretinal/Combined/train/Grade 2/*.*')

G3=glob.glob('/data/neuroretinal/Combined/train/Grade 3/*.*')
G4=glob.glob('/data/neuroretinal/Combined/train/Grade 4/*.*')
Atypical =glob.glob('/data/neuroretinal/Combined/train/Atypical/*.*')




# # print(Normal[0])

for i in range(0,100):
    print(i)
    test_img = Normal[i]
    img = open_image(test_img)
    vn = Normal[i].split('/')[-1][ : -4]
    gcam = GradCam.from_one_img(learn,img)
    gcam.plot(name_img  = vn)
    # plt.imshow(gcam)
    # plt.savefig('/home/g/gv53/tmp/cam.jpg')

    # print('done')



# /home/garima/Desktop/UoL_Projects/6_class_classification/dataset/Combined/train/Normal/


# test_img = Atypical[0]
# img = open_image(test_img)
# vn = Atypical[0].split('/')[-1][ : -4]
# gcam = GradCam.from_one_img(learn,img)
# gcam.plot(name_img  = vn)
    # plt.imshow(gcam)
    # plt.savefig('/home/g/gv53/tmp/cam.jpg')

    # print('done')