import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset



class OCTDataset(Dataset):
    def __init__(self, csv_path, transforms = None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.class2index = {'NORMAL':0, 'DRUSEN':1, 'CNV':2, 'DME':3}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filepath = self.df.iloc[index]['filepath']
        label = self.class2index[self.df.iloc[index]['annotation']]
        image = Image.open(filepath)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


train_data = OCTDataset('./data/training.csv')
val_data = OCTDataset('./data/validation.csv')
test_data = OCTDataset('./data/testing.csv')
