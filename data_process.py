from tools import *
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import numpy as np


def preprocess(data_path, label):
    # 加载数据集
    data = load(data_path)
    # # 查找数据集的空值
    # find_null(data)
    # 删除异常值
    drop_outliers(data)
    # 标准化数据集
    data = standardize_data(data)
    # 采样
    x, y = segment_data(data, seconds=0.2, label=label)
    return x, y


def load_data(path, seed=0):
    # 遍历文件目录中所有csv文件
    x_list = []
    y_list = []

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                print("loading file:{0}".format(file))
                if file.startswith('n'):
                    data_path = os.path.join(root, file)
                    x, y = preprocess(data_path, 0)
                elif file.startswith('ib'):
                    data_path = os.path.join(root, file)
                    x, y = preprocess(data_path, 1)
                elif file.startswith('ob'):
                    data_path = os.path.join(root, file)
                    x, y = preprocess(data_path, 2)
                elif file.startswith('tb'):
                    data_path = os.path.join(root, file)
                    x, y = preprocess(data_path, 3)
                else:
                    continue
            # 把每次遍历得到的x, y存放到列表中
                x_list.extend(x)
                y_list.extend(y)

    # 将数据集转换为数组
    x = np.array(x_list).squeeze()
    y = np.array(y_list)

    # 打乱数据集
    x, y = shuffle(x, y, random_state=seed)

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

    return x_train, x_test, y_train, y_test