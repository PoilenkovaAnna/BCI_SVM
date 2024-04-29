import json
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import torch as nn

def train_test_split(labels, timeseries, test_size):
    indices = np.arange(len(timeseries))
    np.random.shuffle(indices)

    labels = np.array(labels)[indices]
    timeseries =  timeseries[indices]
    train_count = int(len(timeseries) * (1.0 - test_size))

    train_labels, train_timeseries = labels[0:train_count], timeseries[0:train_count]
    test_labels, test_timeseries = labels[train_count:-1], timeseries[train_count:-1]
    return train_labels, train_timeseries, test_labels, test_timeseries

def train_test_split_fix(i_person, labels, person_spectrograms, file_path):
    with open(file_path, 'r') as file:
        python_obj = json.load(file)

    X_train_sectors = python_obj[f'{i_person}']['X_train_sectors']
    X_test_sectors = python_obj[f'{i_person}']['X_test_sectors']
    y_train = python_obj[f'{i_person}']['y_train']
    y_test = np.array(python_obj[f'{i_person}']['y_test'])

    labels = np.array(labels)
    person_spectrograms = np.array(person_spectrograms)

    train_labels = labels[X_train_sectors]
    train_morlets = person_spectrograms[X_train_sectors]
    test_labels = labels[X_test_sectors]
    test_morlets = person_spectrograms[X_test_sectors]
    return train_labels, train_morlets, test_labels, test_morlets

def chunk(indices, chunk_size):
    return nn.split(nn.tensor(indices), chunk_size)


class PSDatasetInx(Dataset):  # Person Spectrograms Dataset than return idx 
    def __init__(self, labels, spectrograms, transform=None, target_transform=None):
        self.labels = labels
        self.spectrograms = spectrograms
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spectrogram, label = self.spectrograms[idx], self.labels[idx]
        if self.transform:
            spectrogram = self.transform(spectrogram)
        if self.target_transform:
            label = self.target_transform(label)
        return spectrogram, label, idx

class PSDataset(Dataset):  # Person Spectrograms Dataset
    def __init__(self, labels, spectrograms, transform=None, target_transform=None):
        self.labels = labels
        self.spectrograms = spectrograms
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spectrogram, label = self.spectrograms[idx], self.labels[idx]
        if self.transform:
            spectrogram = self.transform(spectrogram)
        if self.target_transform:
            label = self.target_transform(label)
        return spectrogram, label


class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.indices = list(range(len(dataset)))
        self.batch_size = batch_size

    def __iter__(self):
        random.shuffle(self.indices)
        batches  = chunk(self.indices, self.batch_size)
        combined = [batch.tolist() for batch in batches]
        return iter(self.indices )

    def delete(self, del_indices):
        self.indices = list(set(self.indices) - set(del_indices))

    def __len__(self):
        return (len(self.indices)) // self.batch_size