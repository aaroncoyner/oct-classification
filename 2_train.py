import os

import numpy as np
import pandas as pd
from PIL import Image
from skorch import NeuralNetClassifier

from skorch.callbacks import LRScheduler, Checkpoint, EpochScoring
from skorch.helper import predefined_split

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms



class OCTDataset(Dataset):
    def __init__(self, csv_path, transform):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.class2index = {'NORMAL':0, 'DRUSEN':1, 'CNV':2, 'DME':3}
        self.labels = np.vectorize(self.class2index.get)(np.array(self.df.loc[:, 'annotation']))
        self.samples = np.array(self.df.loc[:, 'filepath'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filepath = self.df.iloc[index]['filepath']
        label = self.class2index[self.df.iloc[index]['annotation']]
        image = Image.open(filepath).convert('RGB')
        image = self.transform(image)
        return image, label


class PretrainedModel(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_features)
        self.model = model

    def forward(self, x):
        return self.model(x)


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
LR = 0.001
NUM_CLASSES = 4
NUM_WORKERS = 0
NUM_EPOCHS = 10

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
if device == 'cuda':
    torch.cuda.empty_cache()

f_params = './model.pt'
f_history = './history.json'
csv_name = './probability.csv'

train_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.RandomRotation(25),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])

test_transforms = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],
                                                           [0.5, 0.5, 0.5])])

train_data = OCTDataset('./data/training.csv', test_transforms)
val_data = OCTDataset('./data/validation.csv', test_transforms)
test_data = OCTDataset('./data/testing.csv', test_transforms)

labels = train_data.labels
normal_weight = 1 / len(labels[labels == 0])
drusen_weight = 1 / len(labels[labels == 1])
cnv_weight = 1 / len(labels[labels == 2])
dme_weight = 1 / len(labels[labels == 3])

sample_weights = np.array([normal_weight, drusen_weight, cnv_weight, dme_weight])
weights = sample_weights[labels]
sampler = WeightedRandomSampler(weights, len(train_data), replacement=True)

checkpoint = Checkpoint(monitor='valid_loss_best',
                        f_params=f_params,
                        f_history=f_history,
                        f_optimizer=None,
                        f_criterion=None)

train_acc = EpochScoring(scoring='accuracy',
                         on_train=True,
                         name='train_acc',
                         lower_is_better=False)

net = NeuralNetClassifier(PretrainedModel,
                          criterion=nn.CrossEntropyLoss,
                          lr=LR,
                          batch_size=BATCH_SIZE,
                          max_epochs=NUM_EPOCHS,
                          module__output_features=NUM_CLASSES,
                          optimizer=optim.SGD,
                          optimizer__momentum=0.9,
                          iterator_train__num_workers=NUM_WORKERS,
                          iterator_train__sampler=sampler,
                          iterator_valid__shuffle=False,
                          iterator_valid__num_workers=NUM_WORKERS,
                          train_split=predefined_split(val_data),
                          callbacks=[checkpoint, train_acc],
                          device=device)

print()
print(f'Number of classes: {NUM_CLASSES}')
print(f'Number of workers: {NUM_WORKERS}')
print(f'Image size: {IMAGE_SIZE}')
print(f'Number of epochs: {NUM_EPOCHS}')
print(f'Learning rate: {LR}')
print(f'Batch size: {BATCH_SIZE}')
print(f'Device: {device}')
print()
print('Training...')
net.fit(train_data, y=None)

print('\nTesting...')
test_probs = net.predict_proba(test_data)
img_locs = test_data.samples
data = {'img_loc': img_locs,
        'NORMAL': test_probs[:,0],
        'DRUSEN': test_probs[:,1],
        'CNV': test_probs[:,2],
        'DME': test_probs[:,3]}
pd.DataFrame(data=data).to_csv(csv_name, index=False)
