#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision.all import *

import numpy as np
# In[50]:



from fastcore.foundation import L

from fastai.callback.fp16 import to_fp16
from fastai.callback.progress import ProgressCallback
from fastai.callback.schedule import fit_one_cycle

from fastai.data.core import Datasets, show_at
from fastai.data.external import untar_data, URLs
from fastai.data.transforms import IntToFloatTensor, Normalize, ToTensor, IndexSplitter, get_image_files, parent_label, Categorize

from fastai.metrics import accuracy

from fastai.vision.augment import aug_transforms, RandomResizedCrop
from fastai.vision.core import PILImage, imagenet_stats
from fastai.vision.learner import cnn_learner

import random

from sklearn.model_selection import StratifiedKFold

from torchvision.models.resnet import resnet50
print('done')


# In[19]:


train_imgs = get_image_files('/data/neuroretinal/Combined-Binary/train')
tst_imgs = get_image_files('/data/neuroretinal/Combined-Binary/valid')
print(len(train_imgs))
print(len(tst_imgs))
random.shuffle(train_imgs)

start_val = len(train_imgs) - int(len(train_imgs)*.2)
idxs = list(range(start_val, len(train_imgs)))
splitter = IndexSplitter(idxs)
splits = splitter(train_imgs)

split_list = [splits[0], splits[1]]
split_list.append(L(range(len(train_imgs), len(train_imgs)+len(tst_imgs))))
print('This is the split')
print(len(split_list[0]))
print(len(split_list[1]))
split_list


# In[5]:


dsrc = Datasets(train_imgs+tst_imgs, tfms=[[PILImage.create], [parent_label, Categorize]],
                splits = split_list)
show_at(dsrc.train, 3)


# In[ ]:





# In[ ]:





# In[6]:


print(dsrc.n_subsets)

print(len(split_list[2]))


# In[53]:


item_tfms = [ToTensor(), RandomResizedCrop(460, min_scale=0.75, ratio=(1.,1.))]
batch_tfms = [IntToFloatTensor(), *aug_transforms(size=224, max_warp=0), Normalize()]
bs=64

dls = dsrc.dataloaders(bs=bs, after_item=item_tfms, after_batch=batch_tfms)


# In[54]:


dls.show_batch()


# In[14]:


dls.n_subsets


# In[20]:


dls[2].show_batch()


# In[22]:


# learn = cnn_learner(dls, resnet34, pretrained=False, metrics=accuracy).to_fp16()
learn = cnn_learner(dls, resnet50, metrics=[accuracy]).to_fp16()
# learn.fit_one_cycle(1)


# In[42]:


train_labels = L(dsrc.items).map(dsrc.tfms[1])


# In[48]:


kf = StratifiedKFold(n_splits=5, shuffle=False)


# In[ ]:

count = 1
val_pct = []
tst_preds = []
for _, val_idx in kf.split(np.array(train_imgs+tst_imgs), train_labels):
    splits = IndexSplitter(val_idx)
    split = splits(train_imgs)
    split_list = [split[0], split[1]]
    split_list.append(L(range(len(train_imgs), len(train_imgs)+len(tst_imgs))))
    dsrc = Datasets(train_imgs+tst_imgs, tfms=[[PILImage.create], [parent_label, Categorize]],splits=split_list)
    dls = dsrc.dataloaders(bs=bs, after_item=item_tfms, after_batch=batch_tfms)
    learn = cnn_learner(dls, resnet50, metrics=[accuracy])
    learn.fit_one_cycle(30,slice(1e-3,3e-4), pct_start=0.05)
    learn.save('cvalid_binary_'+ str(count))
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    plt.savefig('confusion_matrix_binary' + str(count) +'.png')
    count  = count + 1
    val_pct.append(learn.validate()[1])
    a,b = learn.get_preds(ds_idx=2)
    tst_preds.append(a)


# In[ ]:


# learn.save('cross_valid_6_45')

tst_preds_copy = tst_preds.copy()
accuracy(tst_preds_copy[0], b)
for i in tst_preds_copy:
    print(accuracy(i, b))
    
hat = tst_preds[0]
for pred in tst_preds[1:]:
    hat += pred
    
print('Combined accuracy is ')
print(accuracy(hat, b))
