import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import models, transforms



class OCTDataset(Dataset):
    def __init__(self, csv_path, transform = None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.class2index = {'NORMAL':0, 'DRUSEN':1, 'CNV':2, 'DME':3}
        self.labels = np.vectorize(self.class2index.get)(np.array(self.df.loc[:, 'annotation']))
        # self.labels = [self.class2index[x] for x in np.array(self.df.loc[:, 'annotation'])]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filepath = self.df.iloc[index]['filepath']
        label = self.class2index[self.df.iloc[index]['annotation']]
        image = Image.open(filepath)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


IMAGE_SIZE = (224, 224)


train_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.RandomRotation(25),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5], [0.5])])

test_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5], [0.5])])


train_data = OCTDataset('./data/training.csv', train_transforms)
val_data = OCTDataset('./data/validation.csv')
test_data = OCTDataset('./data/testing.csv')

labels = train_data.labels
normal_weight = 1 / len(labels[labels == 0])
drusen_weight = 1 / len(labels[labels == 1])
cnv_weight = 1 / len(labels[labels == 2])
dme_weight = 1 / len(labels[labels == 3])

sample_weights = np.array([normal_weight, drusen_weight, cnv_weight, dme_weight])
weights = sample_weights[labels]
sampler = WeightedRandomSampler(weights, len(train_data), replacement=True)
