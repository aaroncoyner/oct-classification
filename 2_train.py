import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, Checkpoint, EpochScoring, EarlyStopping
from skorch.helper import predefined_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, WeightedRandomSampler
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


def set_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        torch.cuda.empty_cache()
    return device


def prepare_data(input_dir, random_flips, image_size=(224,224)):
    train_transforms = transforms.Compose([transforms.Resize(image_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomRotation(45),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    test_transforms = transforms.Compose([transforms.Resize(image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    if random_flips:
        train_data = OCTDataset(os.path.join(input_dir, 'training.csv'), train_transforms)
    else:
        train_data = OCTDataset(os.path.join(input_dir, 'training.csv'), test_transforms)
    val_data = OCTDataset(os.path.join(input_dir, 'validation.csv'), test_transforms)
    test_data = OCTDataset(os.path.join(input_dir, 'testing.csv'), test_transforms)
    return train_data, val_data, test_data


def configure_callbacks(stop_early, output_dir):
    f_params = os.path.join(output_dir, 'model.pt')
    f_history = os.path.join(output_dir, 'history.json')
    checkpoint = Checkpoint(monitor='valid_loss_best',
                            f_params=f_params,
                            f_history=f_history,
                            f_optimizer=None,
                            f_criterion=None)
    train_acc = EpochScoring(scoring='accuracy',
                             on_train=True,
                             name='train_acc',
                             lower_is_better=False)
    if stop_early:
        early_stopping = EarlyStopping()
        callbacks = [checkpoint, train_acc, early_stopping]
    else:
        callbacks = [checkpoint, train_acc]
    return callbacks


def configure_sampler(weight_samples):
    if weight_samples:
        labels = train_data.labels
        normal_weight = 1 / len(labels[labels == 0])
        drusen_weight = 1 / len(labels[labels == 1])
        cnv_weight = 1 / len(labels[labels == 2])
        dme_weight = 1 / len(labels[labels == 3])
        sample_weights = np.array([normal_weight, drusen_weight, cnv_weight, dme_weight])
        weights = sample_weights[labels]
        sampler = WeightedRandomSampler(weights, len(train_data), replacement=True)
    else:
        sampler = None
    return sampler


def train(train_data, val_data, random_flips=True, weight_samples=True, stop_early=True,
          output_dir='./', num_classes=4, image_size=(224,224), num_epochs=50, lr=0.001,
          batch_size=32, num_workers=0):
    device = set_device()
    sampler = configure_sampler(weight_samples)
    callbacks = configure_callbacks(stop_early, output_dir)
    print()
    print(f'Number of classes: {num_classes}')
    print(f'Number of workers: {num_workers}')
    print(f'Image size: {image_size}')
    print(f'Number of epochs: {num_epochs}')
    print(f'Learning rate: {lr}')
    print(f'Batch size: {batch_size}')
    print(f'Device: {device}')
    print()
    print('Training...')
    net = NeuralNetClassifier(PretrainedModel,
                              criterion=nn.CrossEntropyLoss,
                              lr=lr,
                              batch_size=batch_size,
                              max_epochs=num_epochs,
                              module__output_features=num_classes,
                              optimizer=optim.SGD,
                              optimizer__momentum=0.9,
                              iterator_train__num_workers=num_workers,
                              iterator_train__sampler=sampler,
                              iterator_valid__shuffle=False,
                              iterator_valid__num_workers=num_workers,
                              train_split=predefined_split(val_data),
                              callbacks=callbacks,
                              device=device)
    net.fit(train_data, y=None)
    return net


def test(net, data, output_dir='./'):
    print()
    print('Testing...')
    probs = net.predict_proba(data)
    img_locs = test_data.samples
    csv_data = {'img_loc': img_locs,
                'NORMAL': probs[:,0],
                'DRUSEN': probs[:,1],
                'CNV': probs[:,2],
                'DME': probs[:,3]}
    csv_name = os.path.join(output_dir, 'probability.csv')
    pd.DataFrame(data=csv_data).to_csv(csv_name, index=False)
    print(f'Done. Outputs written to: {csv_name}')
    print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--input_dir', type=str, default='./data')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--random_flips', type=bool, default=True)
    parser.add_argument('--stop_early', type=bool, default=True)
    parser.add_argument('--weight_samples', type=bool, default=True)
    return parser.parse_args()



if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parse_args()
    image_size = (args.image_size,args.image_size)
    train_data, val_data, test_data = prepare_data(input_dir=args.input_dir,
                                                   random_flips=args.random_flips,
                                                   image_size=image_size)
    net = train(train_data,
                val_data,
                batch_size=args.batch_size,
                image_size=image_size,
                lr=args.learning_rate,
                num_classes=args.num_classes,
                num_epochs=args.num_epochs,
                num_workers=args.num_workers,
                output_dir=args.output_dir,
                random_flips=args.random_flips,
                stop_early=args.stop_early,
                weight_samples=args.weight_samples)

    test(net, test_data, args.output_dir)
