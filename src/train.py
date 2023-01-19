from fastai import *
from fastai.vision import *
from fastai.vision.core import *
from fastai.callback import *
from fastai.vision.all import *


#path of the dataset
path1 = "/data/neuroretinal/Combined/train"


fields = DataBlock(blocks=(ImageBlock, CategoryBlock),
   get_items=get_image_files,
   get_y=parent_label,
   splitter=RandomSplitter(valid_pct=0.2, seed=42),
   item_tfms=RandomResizedCrop(224, min_scale=0.5),
   batch_tfms=aug_transforms())

print('done')

dls = fields.dataloaders(path1)
print(dls.vocab)

#create a cnn pipeline (fastai's default class number is the number of folders present in the train dataset)
learn = vision_learner(dls, models.resnet50, metrics=accuracy)

#uncomment below line if you want to load a pre-trained model (here strict=False means no. of class output can be changed 
# i.e you can load pretrained model for 6 class classification model & use it for 3 class classification)
# learn.load("/home/g/gv53/tmp/stageresnet50-7", strict=False, remove_module=True)

#checks number of classes present
# print(data.classes)

#learning rate finder
learn.lr_find()

#plots the learning rate
learn.recorder.plot()
learn.recorder.plot(return_fig=True)
learn.save("./lr.png")

#train the model for 4 epochs
learn.fit_one_cycle(4)

#unfreeze the model
learn.unfreeze()

#train for 150 epochs with learning rate varying between 1e-3,3e-4
learn.fit_one_cycle(150,slice(1e-3,3e-4), pct_start=0.05)   

#prints final accuracy
print(accuracy(*learn.TTA()))

#saves the model
learn.save('model')

#finds images that were most missclassified
interp = ClassificationInterpretation.from_learner(learn)
losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)

#plots top loss images
interp.plot_top_losses(12, figsize=(15,11))

#save the figure
# plt.savefig('loss.png')

#calculate confusion matrix
interp.plot_confusion_matrix()

#find most confused classes
print(interp.most_confused(min_val=2))

#save confusion matrix
plt.savefig('confusion_matrix.png')