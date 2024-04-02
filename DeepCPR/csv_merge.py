# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 19:47:48 2024

@author: ZNDX001
"""

import pandas as pd
import os


def read_file(file_name):
    """
    read files, just keep 'rt' and 'peak area'.
    """
    if file_name.endswith('.xlsx'):
        file = pd.read_excel(file_name, usecols=['rt', 'peak area'])
    if file_name.endswith('.csv'):
        file = pd.read_csv(file_name, usecols=['rt', 'peak area'])
    return file


def is_close(a, b, tolerance=0.01):
    """Check that the two values are close enough"""
    return abs(a - b) <= tolerance


def can_merge(row, next_row):
    """Check if the current line and the next line can be merged"""
    non_zero_current = row.iloc[1:] != 0
    non_zero_next = next_row.iloc[1:] != 0
    return (non_zero_current != non_zero_next).all()


def merge_rows(row, next_row):
    """Merge rows with less non-zero data into rows with more non-zero data"""
    row_copy = row.copy()
    next_row_copy = next_row.copy()

    non_zero_current = row_copy.iloc[1:] != 0
    non_zero_next = next_row_copy.iloc[1:] != 0
    if non_zero_current.sum() > non_zero_next.sum():
        row_copy.iloc[1:].loc[non_zero_next] = next_row_copy.iloc[1:].loc[non_zero_next]
        return row_copy
    else:
        next_row_copy.iloc[1:].loc[non_zero_current] = row_copy.iloc[1:].loc[non_zero_current]
        return next_row_copy

def merge_dataframes(dfs, file_names):
    """Merge DataFrame"""
    merged = pd.DataFrame()

    # Get all unique rt values, consider proximity values
    rt_values = []
    for df in dfs:
        for rt in df['rt']:
            if not any(is_close(rt, existing_rt) for existing_rt in rt_values):
                rt_values.append(rt)

    # Initialising the structure of the merged DataFrame
    merged['rt'] = sorted(rt_values)
    for i, df in enumerate(dfs):
        merged[file_names[i].split('.')[0]] = 0

    # Merging 'peak area' data
    for i, df in enumerate(dfs):
        for _, row in df.iterrows():
            closest_index = merged['rt'].sub(row['rt']).abs().idxmin()
            if is_close(merged.at[closest_index, 'rt'], row['rt']):
                merged.at[closest_index, file_names[i].split('.')[0]] = row['peak area']

    i = 0
    while i < len(merged) - 2:
        row = merged.iloc[i]
        next_row = merged.iloc[i + 1]

        if can_merge(row, next_row):
            next2_row = merged.iloc[i + 2]
            if can_merge(next_row, next2_row):
                abs_rt_01 = abs(row.iloc[0] - next_row.iloc[0])
                abs_rt_12 = abs(next_row.iloc[0] - next2_row.iloc[0])
                if abs_rt_01 <= abs_rt_12:
                    merged_row = merge_rows(row, next_row)
                    merged.iloc[i] = merged_row
                    merged = merged.drop(merged.index[i + 1])
                else:
                    merged_row = merge_rows(next_row, next2_row)
                    merged.iloc[i + 1] = merged_row
                    merged = merged.drop(merged.index[i + 2])
            else:
                merged_row = merge_rows(row, next_row)
                merged.iloc[i] = merged_row
                merged = merged.drop(merged.index[i + 1])
        else:
            i += 1

    return merged


def peaktable(single_path, save_path):
    """
    Merge all files into one peak table.
    """
    file_names = os.listdir(single_path)
    dataframes = [read_file(single_path + '/' + file_name) for file_name in file_names]

    merged_df = merge_dataframes(dataframes, file_names)
    merged_df = merged_df.T

    merged_df.to_csv(save_path + '/peak_area_table.csv', header=False)