import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


directory_str = './simplified/'

directory = os.fsencode(directory_str)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    ndjson_path = directory_str + filename
    print(ndjson_path)

    df = pd.read_json(ndjson_path, lines=True)

    drawing_array = df[df.recognized == True].drawing.values

    for i in range(10000):
        plt.figure(figsize=[2.24, 2.24])
        plt.axis(False)

        for j in range(len(drawing_array[i])):
            x_coord = drawing_array[i][j][0]
            y_coord = drawing_array[i][j][1]

            x = np.array(x_coord)
            y = np.array(y_coord)

            plt.plot(x, y, c='black')
    
        path = './custom-image-dataset/' + df.word[0] + '-' + str(i) + '.jpg'
        plt.savefig(path)
        plt.clf()