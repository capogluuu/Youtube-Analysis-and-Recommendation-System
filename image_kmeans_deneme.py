import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2
from imageio import imread
from skimage.transform import resize
from sklearn.cluster import KMeans
from matplotlib.colors import to_hex

np.random.seed(0)
filepath = 'images/art00270005.png'

img = imread(filepath)

# Show image
plt.axis('off')
plt.imshow(img);
img = resize(img, (200, 200))
data = pd.DataFrame(img.reshape(-1, 3),
                    columns=['R', 'G', 'B'])

kmeans = KMeans(n_clusters=6,
                random_state=0)
data['Cluster'] = kmeans.fit_predict(data)
palette = kmeans.cluster_centers_
palette_list = list()
for color in palette:
	palette_list.append([[tuple(color)]])


for color in palette_list:
	print(to_hex(color[0][0]))
	plt.figure(figsize=(1, 1))
	plt.axis('off')
	plt.imshow(color);
	#plt.show();

plt.show()
data['R_cluster'] = data['Cluster'].apply(lambda x: palette_list[x][0][0][0])
data['G_cluster'] = data['Cluster'].apply(lambda x: palette_list[x][0][0][1])
data['B_cluster'] = data['Cluster'].apply(lambda x: palette_list[x][0][0][2])
img_c = data[['R_cluster', 'G_cluster', 'B_cluster']].values
img_c = img_c.reshape(200, 200, 3)
img_c = resize(img_c, (800, 1200))
plt.axis('off')
plt.imshow(img_c)
plt.show()

print(palette_list)