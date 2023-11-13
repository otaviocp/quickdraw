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
from torch.utils.data import Dataset, DataLoader, random_split

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
    def __init__(self, num_classes, file_list, directory_str, transform=None):
        self.directory_str = directory_str
        self.num_classes = num_classes
        self.transform = transform

        # Get random index to load random files
        self.file_it = random.sample(range(0, len(file_list)), num_classes)

        # Get labels list
        custom_label_list = []
        for i in self.file_it:
            custom_label_list.append(file_list[i].removesuffix('.npy'))

        self.label_list = custom_label_list

        self.x = []
        self.y = []
        cont = 0

        for data_file in custom_label_list:
            path = directory_str + data_file + '.npy'
            classes = np.load(path).astype(np.float32)

            self.y += [cont]*len(classes)
            cont += 1

            for i in range(len(classes)):
                self.x.append(classes[i])

        self.x = np.array(self.x)
        self.y= np.array(self.y).reshape(-1, 1)
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = self.x[idx], self.y[idx]
            
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

num_classes = 10
num_epochs = 100
batch_size = 400
learning_rate = 0.01

dataset = QuickDraw(num_classes, file_list, directory_str, transform=ToTensor())
train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], torch.Generator().manual_seed(42))
print(len(train_dataset), len(test_dataset))
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

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

    flag = 0
    for images, labels in test_loader:
        outputs = model(images)
        labels = labels.squeeze()
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        evaluate = (predicted == labels).float()
        n_correct += evaluate.sum().item()

        if flag == 0:
            classes = dataset.get_label_list()
            for i in range(6):
                plt.subplot(2, 3, i+1)
                plt.imshow(images[i].squeeze(), cmap='gray')
                plt.title(classes[predicted[i]])
                plt.axis("off")
            
            flag = 1
            plt.show()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100 * n_correct / n_samples
    print(f'Accuracy = {acc} %')

    classes = dataset.get_label_list()
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc:.2f} %')

    # examples = iter(test_loader)
    # samples, labels = next(examples)

    # for i in range(6):
    #     plt.subplot(2, 3, i+1)5
    #     plt.imshow(samples[i], cmap='gray')
    #     plt.title(train_ds.class_names[labels[i]])
    #     plt.axis("off")

    # plt.show()
