import csv
from tqdm import tqdm
data = []
with open('output.tsv') as f:
    freader = csv.reader(f, delimiter = '\t')
    for row in tqdm(freader):
        for num in row:
            num = float(num)
        data.append(row)

color_label = []
with open('output_label.tsv') as f:
    temp = f.read().split('\n')[1:-1]
    for row in temp:
        color_label.append(float(row.split('\t')[-1]))
#         print(row.split('\t')[-1], end = '')

from sklearn.manifold import TSNE

# model = TSNE(n_components = 2, random_state = 0)
# data_1k = data[:1000]
# color_label_1k = color_label[:1000]
# tsne_data_1k = model.fit_transform(data_1k)

import numpy as np
import pandas as pd

# tsne_data_1k = np.vstack((tsne_data_1k.T, color_label_1k)).T
# print(tsne_data_1k.shape)
# tsne_df_1k = pd.DataFrame(data = tsne_data_1k, columns = ('Dim1', 'Dim2', 'difficulty'))
# fig = tsne_df_1k.plot.scatter(x='Dim1', y = 'Dim2', s=2, c = 'difficulty', colormap = 'viridis').get_figure()
# fig.set_figheight(15)
# fig.set_figwidth(15)
# fig.savefig('tsne_1k.png')

# model = TSNE(n_components = 2, random_state = 0)
# data_10k = data[:10000]
# color_label_10k = color_label[:10000]
# tsne_data_10k = model.fit_transform(data_10k)
# tsne_data_10k = np.vstack((tsne_data_10k.T, color_label_10k)).T
# print(tsne_data_10k.shape)
# tsne_df_10k = pd.DataFrame(data = tsne_data_10k, columns = ('Dim1', 'Dim2', 'difficulty'))
# fig = tsne_df_10k.plot.scatter(x='Dim1', y = 'Dim2', s=2, c = 'difficulty', colormap = 'viridis').get_figure()
# fig.set_figheight(15)
# fig.set_figwidth(15)
# fig.savefig('tsne_10k.png')

# model = TSNE(n_components = 2, random_state = 0)
# data_50k = data[:50000]
# color_label_50k = color_label[:50000]
# tsne_data_50k = model.fit_transform(data_50k)
# tsne_data_50k = np.vstack((tsne_data_50k.T, color_label_50k)).T
# print(tsne_data_50k.shape)
# tsne_df_50k = pd.DataFrame(data = tsne_data_50k, columns = ('Dim1', 'Dim2', 'difficulty'))
# fig = tsne_df_50k.plot.scatter(x='Dim1', y = 'Dim2', s=2, c = 'difficulty', colormap = 'viridis').get_figure()
# fig.set_figheight(15)
# fig.set_figwidth(15)
# fig.savefig('tsne_50k.png')

model = TSNE(n_components = 2, random_state = 0)
tsne_data = model.fit_transform(data)
tsne_data = np.vstack((tsne_data.T, color_label)).T
print(tsne_data.shape)
tsne_df = pd.DataFrame(data = tsne_data, columns = ('Dim1', 'Dim2', 'difficulty'))
fig = tsne_df.plot.scatter(x='Dim1', y = 'Dim2', s=0.5, c = 'difficulty', colormap = 'viridis').get_figure()
fig.set_figheight(15)
fig.set_figwidth(15)
fig.savefig('tsne.png')