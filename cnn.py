import random
import os
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

directory_str = './dataset/'
directory = os.fsencode(directory_str)
file_list = []

# Get all dataset files
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    file_list.append(filename)

print(len(file_list))

# Define the QuickDraw dataset class
class QuickDraw(Dataset):
    def __init__(self, num_classes, file_list, directory_str, train, transform=None):
        self.directory_str = directory_str
        self.num_classes = num_classes
        self.transform = transform
        self.train = train
        # Get random index to load random files
        self.file_it = random.sample(range(0, len(file_list)), num_classes)

        # Get labels list
        custom_label_list = []
        for i in self.file_it:
            custom_label_list.append(file_list[i].removesuffix('.npy'))

        self.labels = custom_label_list

        # Read data files and create an array
        train_data_list = []
        test_data_list = []

        for data_file in custom_label_list:
            path = directory_str + data_file + '.npy'
            classes = np.load(path)

            train_size = int(round(0.8*len(classes)))
            test_size = int(round(0.2*len(classes)))

            for i in range(train_size):
                train_data_list.append(classes[i])
            for i in range(test_size):
                j = train_size+i
                test_data_list.append(classes[j])
        
        self.train_data = np.array(train_data_list)
        self.test_data = np.array(test_data_list)


    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


    def __getitem__(self, idx):
        if self.train:
            label = np.array(int(idx/len(self.train_data)))
            sample = self.train_data[idx], label

        else:
            label = np.array(int(idx/len(self.test_data)))
            sample = self.test_data[idx], label
            
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor():
    def __call__(self, sample):
        inputs, targets = sample

        inputs = torch.from_numpy(inputs)

        return inputs.view(28,28), torch.from_numpy(targets)

# Hyper parameters
num_epochs = 4
batch_size = 50
learning_rate = 0.001

train_data = QuickDraw(10, file_list, directory_str, train=True, transform=ToTensor())
test_data = QuickDraw(10, file_list, directory_str, train=False, transform=ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

examples = iter(train_loader)
samples, labels = next(examples)

print(samples.shape)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i], cmap='gray')

plt.show()