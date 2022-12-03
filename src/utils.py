import os
import shutil
import glob
import json

import numpy as np
from tqdm import tqdm
from constants import *
from PIL import Image
import torchvision.transforms as transforms
import transforms as T
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
class CityscapesInstanceSegProvisioner:
    def __init__(self, root_path
                 , instance_classes = DEFAULT_INSTANCE_CLASSES
                 ,output_size = DEFAULT_OUTPUT_SIZE
                 ,object_minimal_area=DEFAULT_MINIMAL_AREA
                 , minimal_number_of_valid_objs=DEFAULT_MINIMAL_OBJECT_NUMBER):
        self.root_path = root_path
        self.input_path_mask = os.path.join(root_path, CITYSCAPES, GT_FINE)
        self.input_paths_img = os.path.join(root_path, CITYSCAPES, LEFT_IMG_8_BIT)
        self.output_path_mask = os.path.join(root_path, FINAL,INSTANCES, GT_FINE)
        self.output_path_img = os.path.join(root_path, FINAL,INSTANCES, LEFT_IMG_8_BIT)
        self.instance_classes = instance_classes
        self.output_size = output_size
        self.object_minimal_area = object_minimal_area
        self.minimal_number_of_valid_objs = minimal_number_of_valid_objs


    def summary(self):
        masks = len(glob.glob(f"{self.output_path_mask}/*.png"))
        images = len(glob.glob(f"{self.output_path_img}/*.png"))
        print(f"There has been {masks} masks and {images} annotated ")
    def prepare_workspace(self):
        if os.path.exists(self.input_path_mask) and os.path.exists(self.input_paths_img):

            if os.path.exists(self.output_path_mask) or os.path.exists(self.output_path_img):
                print("DELETING PREVIOUS RESULTS")
                shutil.rmtree(os.path.join(self.root_path, FINAL))
            os.makedirs(self.output_path_mask, True)
            os.makedirs(self.output_path_img, True)


        else:
            raise Exception("There is no input data")
        print(
            f"CONFIG:\n instance_classes: {self.instance_classes} \n object_minimal_area:{self.object_minimal_area}\n minimal_number_of_valid_objects:{self.minimal_number_of_valid_objs}")

    def copy_valid_files(self):
        annotation_files_to_verify = sorted(glob.glob(f"{self.input_path_mask}/*/*.json"))[0:1000]

        for file in tqdm(annotation_files_to_verify):

            file_name = file.split("\\")[-1]

            valid_file, valid_objs_number = self.validate_file(file_path=file)
            if valid_objs_number >= self.minimal_number_of_valid_objs:
                final_path = f"{self.output_path_mask}/{file_name}"
                with open(final_path, "w") as outfile:
                    json.dump(valid_file, outfile, indent=4)
            else:
                continue
        print(f"Total number of obtained valid files: {len(os.listdir(self.output_path_mask))}")

    def delete_json(self):
        files_to_delete = glob.glob(f"{self.output_path_mask}/*.json")
        print("Deleting JSON files")
        for file in files_to_delete:
            os.remove(file)

    def copy_valid_images(self):
        files_to_copy = os.listdir(self.output_path_mask)
        img_with_source_and_destination = sorted([self.modify_path(file_name) for file_name in files_to_copy])
        print("Copying images of valid masks")
        for source, destination in tqdm(img_with_source_and_destination):
            shutil.copyfile(source, destination)

    def resize_masks(self):
        masks = sorted(glob.glob(f"{self.output_path_mask}/*.png"))
        print("Resizing masks")
        for mask_path in tqdm(masks):
            self.resize(mask_path)

    def validate_resized_masks(self):
        masks = sorted(glob.glob(f"{self.output_path_mask}/*.png"))
        print("Validating resized masks")
        invalid_masks= []
        for path in masks:
            mask = Image.open(path)
            mask = np.array(mask)
            obj_ids =np.array([obj for obj in np.unique(mask) if obj >1000])
            masks=(mask == obj_ids[:,None,None])
            obj_num = len(obj_ids)
            for i in range(obj_num):
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if not(xmin != xmax != ymin != ymax):
                    if(path not in invalid_masks):
                        invalid_masks.append(path)

        for mask_path in invalid_masks:
            print(mask_path)
            # os.remove(mask_path)


    def resize_images(self):
        images = sorted(glob.glob(f"{self.output_path_img}/*.png"))
        print("Resizing images")
        for img_path in tqdm(images):
            self.resize(img_path)

    def resize(self,path):
        img = Image.open(path)
        re_sized = img.resize(DEFAULT_OUTPUT_SIZE, resample=Image.Resampling.NEAREST)
        re_sized.save(path)



    def modify_path(self, file_name):
        file_name = file_name.replace("_gtFine_instanceTrainIds", "_leftImg8bit")
        input_path = glob.glob(f"{self.input_paths_img}/*/{file_name}")[0]
        final_path = os.path.join(self.output_path_img, file_name)
        return input_path, final_path

    def validate_file(self, file_path):
        valid_objs = []
        file = json.load(open(file_path))
        objects = file["objects"]
        for obj in objects:
            label = obj["label"]
            area = self._calculate_area(obj["polygon"])
            if label in self.instance_classes and area > self.object_minimal_area:
                valid_objs.append(obj)

        file["objects"] = valid_objs

        return (file, len(valid_objs))

    # def validate_after_resize(self):



    def _calculate_area(self, polygon_cords):

        x_cords = [int(cords[0]) for cords in polygon_cords]
        y_cords = [int(cords[1]) for cords in polygon_cords]
        x_min = min(x_cords)
        x_max = max(x_cords)
        y_min = min(y_cords)
        y_max = max(y_cords)
        area = (int(x_max) - int(x_min)) * (int(y_max) - int(y_min))
        return area
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
def display_segmentation_mask(img,masks,labels):

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    labels_encoded = [dict(ids_to_labels.values())[label] for label in labels.tolist()]
    img_tensor=transform(img)
    b_bs= masks_to_boxes(masks)
    img_with_masks = draw_segmentation_masks(img_tensor, masks, alpha=0.8)
    img_wit_masks_and_bb = draw_bounding_boxes(img_with_masks,b_bs,colors="white",width=1,labels=labels_encoded)
    plt.figure(figsize=(20,20))
    plt.imshow(F.to_pil_image(img_wit_masks_and_bb))
def collate_fn(batch):
    return tuple(zip(*batch))