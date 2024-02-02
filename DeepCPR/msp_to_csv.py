# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:15:36 2024

@author: ZNDX001
"""
import pandas as pd


def msp_to_csv(msp_file, csv_save_path, csv_save_name):
    with open(msp_file, 'r') as file:
        lines = file.readlines()

    # 解析数据
    peaks = []
    for line in lines:
        if line.startswith('Name') or line.startswith('Num Peaks'):
            continue
        parts = line.split()
        if len(parts) == 2:
            m_z, intensity = parts
            peaks.append({'m/z': float(m_z), 'intensity': float(intensity)})

    # 转换为 DataFrame
    df = pd.DataFrame(peaks)
    df.to_csv(csv_save_path + '/' + csv_save_name, index=False, header=False)