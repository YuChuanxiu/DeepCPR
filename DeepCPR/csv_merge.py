# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 19:47:48 2024

@author: ZNDX001
"""

import pandas as pd
import os


def read_file(file_name):
    """读取文件，只保留 'rt' 和 'peak area' 列"""
    if file_name.endswith('.xlsx'):
        file = pd.read_excel(file_name, usecols=['rt', 'peak area'])
    if file_name.endswith('.csv'):
        file = pd.read_csv(file_name, usecols=['rt', 'peak area'])
    return file


def is_close(a, b, tolerance=0.01):
    """检查两个数值是否足够接近"""
    return abs(a - b) <= tolerance


def can_merge(row, next_row):
    """检查当前行和下一行是否可以合并"""
    non_zero_current = row.iloc[1:] != 0
    non_zero_next = next_row.iloc[1:] != 0
    return (non_zero_current != non_zero_next).all()


def merge_rows(row, next_row):
    """将非零数据少的行合并到非零数据多的行"""
    # 创建副本以避免 SettingWithCopyWarning
    row_copy = row.copy()
    next_row_copy = next_row.copy()

    non_zero_current = row_copy.iloc[1:] != 0
    non_zero_next = next_row_copy.iloc[1:] != 0
    if non_zero_current.sum() > non_zero_next.sum():
        # 选择性更新副本
        row_copy.iloc[1:].loc[non_zero_next] = next_row_copy.iloc[1:].loc[non_zero_next]
        return row_copy
    else:
        next_row_copy.iloc[1:].loc[non_zero_current] = row_copy.iloc[1:].loc[non_zero_current]
        return next_row_copy

def merge_dataframes(dfs, file_names):
    """合并DataFrame列表"""
    merged = pd.DataFrame()

    # 获取所有唯一的rt值，考虑接近值
    rt_values = []
    for df in dfs:
        for rt in df['rt']:
            if not any(is_close(rt, existing_rt) for existing_rt in rt_values):
                rt_values.append(rt)

    # 初始化合并后DataFrame的结构
    merged['rt'] = sorted(rt_values)
    for i, df in enumerate(dfs):
        merged[file_names[i].split('.')[0]] = 0

    # 合并'peak area'数据
    for i, df in enumerate(dfs):
        for _, row in df.iterrows():
            # 找到最接近的rt值的索引
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
    file_names = os.listdir(single_path)
    dataframes = [read_file(single_path + '/' + file_name) for file_name in file_names]

    merged_df = merge_dataframes(dataframes, file_names)
    merged_df = merged_df.T

    merged_df.to_csv(save_path + '/peak_area_table.csv', header=False)



# =============================================================================
# # CSV文件路径列表
# single_path = 'C:/DeepEER2/LMJdata/23_04_18-24-315_zhisone/single'
# save_path = 'C:/DeepEER2/LMJdata/23_04_18-24-315_zhisone'
# file_names = os.listdir(single_path)
# 
# # 读取CSV文件
# dataframes = [read_file(single_path + '/' + file_name) for file_name in file_names]
# 
# # 合并DataFrame
# merged_df = merge_dataframes(dataframes, file_names)
# merged_df = merged_df.T
# 
# # 保存合并后的DataFrame到新的CSV文件
# merged_df.to_csv(save_path + '/merged_peak_areas_11.csv', header=False)
# =============================================================================

