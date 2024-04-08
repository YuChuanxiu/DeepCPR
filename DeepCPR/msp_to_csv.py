# -*- coding: utf-8 -*-

import pandas as pd
import os


def msp_to_csv(msp_file, csv_save_path, csv_save_name):
    with open(msp_file, 'r') as file:
        lines = file.readlines()

    # loading data
    peaks = []
    for line in lines:
        if line.startswith('Name') or line.startswith('RT') or line.startswith('Num Peaks'):
            continue
        if line.startswith('InChIKey') or line.startswith('Synon') or line.startswith('Retention_index') or line.startswith('Formula') or line.startswith('MW'):
            continue
        if line.startswith('ExactMass') or line.startswith('CAS#') or line.startswith('DB#') or line.startswith('Comments'):
            continue
        parts = line.split()
        if len(parts) == 2:
            m_z, intensity = parts
            peaks.append({'m/z': float(m_z), 'intensity': float(intensity)})

    # transform to DataFrame
    df = pd.DataFrame(peaks)
    df.to_csv(csv_save_path + '/' + csv_save_name, index=False, header=False)
