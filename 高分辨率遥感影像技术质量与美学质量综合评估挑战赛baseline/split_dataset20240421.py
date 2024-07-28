import os
import random



def split_dataset(file_path, train_ratio=0.8, seed=42):
    random.seed(seed)

    with open(file_path,'r') as f:
        lines = f.readlines()

    random.shuffle(lines)

    split_point = int(len(lines)* train_ratio)

    train_lines =  lines[:split_point]
    val_lines = lines[split_point:]

    with open('train.txt', 'w') as train_file:
        train_file.writelines(train_lines)

    with open('val.txt', 'w') as val_file:
        val_file.writelines(val_lines)

file_path = '/media/gy/study/bisai/竞赛/评价/训练集/label.txt'
split_dataset(file_path, train_ratio=0.8,seed=42)