# %%
import numpy as np
import matplotlib.pyplot as plt
import os

directory_str = '../numpy_bitmap/'

directory = os.fsencode(directory_str)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    path = directory_str + filename
    dataset = np.load(path)

    np.random.shuffle(dataset)
    dataset_snippet = dataset[:1000]

    target_path = './dataset/' + filename
    np.save(target_path, dataset_snippet)


# dataset = np.load('../numpy_bitmap/airplane.npy')
# print(dataset.shape)

# np.random.shuffle(dataset)

# dataset_snippet = dataset[:1000]

# np.save('./dataset/airplane.npy', dataset_snippet)

# airplane = np.load('./dataset/airplane.npy')

# test = airplane[1].reshape(28,28)
# print(test.shape)
# plt.imshow(test, cmap='gray')
# plt.show()