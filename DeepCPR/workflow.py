# -*- coding: utf-8 -*-

from DeepCPR import data_resolution
from csv_merge import peaktable
from OPLSDA import OPLS, scatter_cluster, vip_objection, heatmap, permutation_test
import numpy as np
import time
import os
import pandas as pd


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

    OPLS_operation = input("Do you want to execute OPLS-DA? (Category label is required) (yes/no): ")
    if OPLS_operation.lower() in ['yes', 'y']:
        OPLS_op = True
        print("PLease input category label")
        label_type = input("Please choose the way to input label: 1. provide the whole label directly; 2. tell me the nummber of first few samples belong to the same category. please answer '1' or '2':")
        if label_type == 1:
            Y = input("Please provide dataset label(two classes, a list, Using 1 and -1 to distinguish between positive and negative samples): ")
        elif label_type == 2:
            catnumber = input("Please input the number of the same category:")
        else:
            print("Invalid response. Please answer '1' or '2'.")
    elif OPLS_operation.lower() in ['no', 'n']:
        OPLS_op = None
    else:
        print("Invalid response. Please answer 'yes' or 'no'.")

    # OPLS-DA
    if OPLS_op is True:
        opls_savepath = save_path + "/OPLSDA"
        if not os.path.exists(opls_savepath):
            os.makedirs(opls_savepath)

        peak_table_data = pd.read_csv(save_path + '/peak_area_table.csv', index_col='rt')
        
        COMs = peak_table_data.columns.values.astype('float')
        
        data = peak_table_data.values
        data = data.astype(float)
        origin_data = np.copy(data)
        
        if label_type == 1:
            target = Y
        if label_type == 2:
            target = np.zeros((data.shape[0], 1))
            target[0:catnumber, 0] = 1
            target[catnumber:, 0] = -1

        modelDF, summaryDF, VIP_df, xcvTraMN, toVn, tVn = OPLS(data, target, 10)

        scatter_cluster(peak_table_data, target, toVn, tVn, opls_savepath)
        VIP, COM = vip_objection(data, VIP_df, COMs, opls_savepath)
        heatmap(origin_data, VIP, VIP_df, COM, opls_savepath)
        permutation_test(opls_savepath, target)

        # compounds identification
        file_names = os.listdir(save_path + '/single')
        # Storing Matching Results
        matches = {rt: None for rt in COM}
        for rt in COM:
            for file_name in file_names:
                df = pd.read_csv(save_path + '/single/' + file_name)
        
                # Check for the presence of the target rt value in the rt column
                rt_index = df[df['rt'] == rt].index
                if not rt_index.empty:
                    # Record matching filenames and indexes
                    matches[rt] = (file_name, rt_index[0])
                    break

        with open(opls_savepath + '/vip_components_MS_location.txt', 'w') as file:
            for rt, match in matches.items():
                if match:
                    print(f"The index of rt value {rt} in file {match[0]} is {match[1]}", file=file)
                else:
                    print(f"rt value {rt} is not found", file=file)
        

        


















