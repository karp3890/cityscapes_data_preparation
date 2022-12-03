from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image
import glob


class Cityscapes(Dataset):
    def __init__(self,root,ids_to_labels,transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(glob.glob(f"{root}/leftImg8bit/*.png")))
        self.masks = list(sorted(glob.glob(f"{root}/gtFine/*.png")))
        self.ids_to_labels = ids_to_labels


    def __getitem__(self,idx):

        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        img =Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask,dtype="int64")
        obj_ids =np.array([obj for obj in np.unique(mask) if obj >1000])
        masks=(mask == obj_ids[:,None,None])

        obj_num = len(obj_ids)
        boxes = []
        labels = []
        for obj in obj_ids:
            obj=str(obj)[0:2]
            labels.append(self.ids_to_labels[obj][0])
        for i in range(obj_num):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            area = (xmax - xmin)*(ymax-ymin)

            boxes.append([xmin, ymin, xmax, ymax])


        boxes = torch.as_tensor(boxes,dtype=torch.float32)
        labels =torch.as_tensor(labels,dtype=torch.int64)
        masks = torch.as_tensor(masks,dtype=torch.bool)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((obj_num,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target





    def __len__(self):
        return len(self.imgs)
