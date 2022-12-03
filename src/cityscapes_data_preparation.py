import argparse
import datetime
from constants import *
from utils import CityscapesInstanceSegProvisioner
from cityscapesscripts.preparation import createTrainIdInstanceImgs

import shutil
def main(args):
    provider = CityscapesInstanceSegProvisioner(root_path=args.root_path, object_minimal_area=500,
                                                minimal_number_of_valid_objs=2)
    provider.prepare_workspace()
    provider.copy_valid_files()
    createTrainIdInstanceImgs.main(args.final_path)
    provider.delete_json()
    # provider.resize_masks()
    # provider.validate_resized_masks()
    provider.copy_valid_images()
    # provider.resize_images()
    provider.summary()
    shutil.make_archive(args.final_file_name,"zip",f"{args.root_path}\{FINAL}\{INSTANCES}")


if __name__ == "__main__":
    date =datetime.datetime.now().strftime("%d-%m-%Y")
    parser = argparse.ArgumentParser(description="Extract_JSON", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_path', type=str, required=False,
                        default="C:\projects\cityscapes_data_preparation\data")
    parser.add_argument('--final_path', type=str, required=False,
                        default="C:\projects\cityscapes_data_preparation\data\\final\\instances")
    parser.add_argument('--final_file_name', type=str, required=False,
                        default=f"dataset_1024x512_{date}")

    main(parser.parse_args())
