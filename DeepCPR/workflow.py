# -*- coding: utf-8 -*-

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DeepCPR import data_resolution
from DeepCPR.csv_merge import peaktable


if __name__ == "__main__":

    dataset_path = ""
    save_path = ''
    DeepCS_path = 'example/DeepCS.h5'
    DeepCPR_path = 'example/DeepCPR.h5'

    generate_image = input("Do you want to generate the images of resolution? Warning: images generation will spend more time. (yes/no): ")
    if generate_image.lower() in ['yes', 'y']:
        generate_image = True
    elif generate_image.lower() in ['no', 'n']:
        generate_image = None
    else:
        print("Invalid response. Please answer 'yes' or 'no'.")

    # DeepCPR
    data_resolution(dataset_path, DeepCS_path, DeepCPR_path, save_path, generate_image)

    # create peaktable
    peaktable(save_path + '/single', save_path)
    print("All data has been resolved")
        

        


















