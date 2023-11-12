import random
import os
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optmin
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Device  configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        self.label_list = custom_label_list

        # Read data files and create an array
        train_data_list = []
        test_data_list = []
        self.y_train = []
        self.y_test = []

        cont = 0
        for data_file in custom_label_list:
            path = directory_str + data_file + '.npy'
            classes = np.load(path).astype(np.float32)

            train_size = int(round(0.8*len(classes)))
            test_size = int(round(0.2*len(classes)))

            self.y_train += [cont]*train_size
            self.y_test += [cont]*test_size

            cont += 1

            for i in range(train_size):
                train_data_list.append(classes[i])
            for i in range(test_size):
                j = train_size+i
                test_data_list.append(classes[j])
        
        self.train_data = np.array(train_data_list)
        self.test_data = np.array(test_data_list)
        self.y_train = np.array(self.y_train).reshape(-1, 1)
        self.y_test = np.array(self.y_test).reshape(-1, 1)
        

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


    def __getitem__(self, idx):
        if self.train:
            sample = self.train_data[idx], self.y_train[idx]

        else:
            sample = self.test_data[idx], self.y_test[idx]
            
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_label_list(self):
        return self.label_list

class ToTensor():
    def __call__(self, sample):
        inputs, targets = sample

        inputs = torch.from_numpy(inputs)
        inputs = inputs.view(28,28)
        inputs = inputs.unsqueeze(0)
        targets = torch.from_numpy(targets)

        return inputs, targets

class CNN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Hyper parameters

num_classes=10
num_epochs = 100
batch_size = 400
learning_rate = 0.01

train_data = QuickDraw(num_classes, file_list, directory_str, train=True, transform=ToTensor())
test_data = QuickDraw(num_classes, file_list, directory_str, train=False, transform=ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# examples = iter(train_loader)
# samples, labels = next(examples)

#print(samples.shape)
# for i in range(6):
#     plt.subplot(2, 3, i+1)5
#     plt.imshow(samples[i], cmap='gray')

# plt.show()

model = CNN_Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        images, labels = data
        labels = labels.squeeze()
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(num_classes)]
    n_class_samples = [0 for i in range(num_classes)]
    for images, labels in test_loader:
        outputs = model(images)
        labels = labels.squeeze()
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        evaluate = (predicted == labels).float()
        n_correct += evaluate.sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100 * n_correct / n_samples
    print(f'Accuracy = {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')