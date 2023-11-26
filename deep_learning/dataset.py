from torch.utils.data import DataLoader, Dataset
import numpy as np


# 定义Dataset, DataLoader
class SequenceDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data


    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y

class PredictDataset(Dataset):
    def __init__(self, x_data):
        self.x_data = x_data


    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        return x


# 创建Dataset实例#, 创建DataLoader实例
def create_dataloader(x_train, x_test, y_train, y_test, args):
    train_dataset = SequenceDataset(x_train, y_train)
    test_dataset = SequenceDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True)
    valid_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False)
    return train_dataloader, valid_dataloader


def create_predict_dataloader(x_test):
    test_dataset = PredictDataset(x_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_dataloader