import os

from Cityscapes import Cityscapes
from constants import *
import shutil
import glob
path = "C:\projects\cityscapes_data_preparation\data\\final\instances"
dataset = Cityscapes("C:\projects\cityscapes_data_preparation\data\\final\instances", ids_to_labels)
invalid_images=[]
for i in range(0,len(dataset)):
    img,target = dataset.__getitem__(i)
    for box in target["boxes"]:
        if(box[0]==box[2] or box[1]==box[3]):
            invalid_images.append(i)
print(invalid_images)
