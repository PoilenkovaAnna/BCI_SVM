import sys
sys.path.insert(0,'/content/gdrive/MyDrive/Colab_Notebooks/ipp-clf-eeg/classification/')

import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

import torchvision
from torchvision import transforms

from data_preparation.selection_and_segmentation import get_person_segments, normalize_labels
from conversion_to_spectrograms.wavelet_transforms import convert_person_segments_to_spectrograms, normalize_spectrograms
from datasets.inference import train_test_split_fix, train_test_split
from datasets.inference import PSDataset, BatchSampler
from datasets.transforms import ToTensor, NoiseTransform


def made_dataset_for_one_person_MULTclassif(edf_person, i_person, test_config, data_config):
    tc = test_config

    new_size = (128, 512)

    segments, labels = get_person_segments(edf_person, tc.target_channel_sets)
    labels = normalize_labels(labels, 1 )

   # new_segments = DWT(segments)

    #segments = make_segments_centered(new_segments)[0:target_channels] # ЭМГ - индикатор

    person_spectrograms = convert_person_segments_to_spectrograms(segments, test_config )
    person_spectrograms = normalize_spectrograms(person_spectrograms)

    # Set seed
    np.random.seed(tc.seed)
    torch.manual_seed(tc.seed)

    # Split
    train_labels, train_morlets, test_labels, test_morlets = train_test_split_fix(i_person,labels, person_spectrograms, data_config.file_path )

    train_transform = transforms.Compose([
        #NoiseTransform(test_config.noise_transform_scale),
        ToTensor(),
        #torchvision.transforms.Resize(new_size, torchvision.transforms.InterpolationMode.BICUBIC ),
        transforms.GaussianBlur(kernel_size = 5, sigma=(0.1, 0.3))

    ])

    test_transform = transforms.Compose([
        ToTensor(),
        #torchvision.transforms.Resize(new_size, torchvision.transforms.InterpolationMode.BICUBIC ),

    ])

    #print(f'Размер входных данных - {len(train_morlets[0])}, { len(train_morlets[0][0])},  { len(train_morlets[0][0][0])}')
    train_data = PSDataset(train_labels, train_morlets, transform = train_transform)
    test_data = PSDataset(test_labels, test_morlets, transform = test_transform)

    custom_batch_sampler = BatchSampler(train_data, tc.batch_size )

    custom_train_loader = DataLoader(train_data, sampler = custom_batch_sampler, batch_size = tc.batch_size)
    train_loader = DataLoader(train_data, batch_size = tc.batch_size, shuffle = True )
    test_loader = DataLoader(test_data, shuffle=False)

    #print(f'Испытуемый - {edf_person}')
    #print(f'Всего данных - {len(person_spectrograms)}, test - {len(train_data)} train - {len(test_data)}')

    return train_loader, test_loader, custom_batch_sampler