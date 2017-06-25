print(__doc__)

import csv
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

#IN

input_data = []
static_columns = []
clusters_count = 3

# INPUT

with open('dgma1.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        row = [float(i.replace(",",".")) for i in row]
        static_columns.append([row[0], row[1]])
        input_data.append(row)


input_data = np.array(input_data)
performed_data = input_data[:,2].reshape(-1,1)

fig = plt.figure(figsize=(12, 8))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

ax.scatter(input_data[:,0], input_data[:,1], input_data[:,2], c='black')


# CALCULATION:

kmeans = KMeans(n_clusters = clusters_count)
kmeans.fit(performed_data)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# OUTPUT:

output_data = np.c_[labels, performed_data, input_data]
output_data = output_data[np.argsort(output_data[:,0])]
output_data = [np.delete(i,4) for i in output_data]


clusters = {}

sort_labels = labels[np.argsort(labels)]

l_list = []

for i in np.arange(clusters_count):
    l_list.append(list(sort_labels).index(i))

l_list.pop(0)

with open('output.csv', 'wb') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(["Centroids:"])
    for row in centroids:
        filewriter.writerow(row)
    filewriter.writerow(["Clasters:"])
    for i in np.split(output_data, l_list):
        filewriter.writerow(["Max:", "Min:", "Ave:"])
        maxi = np.max(i[:,1])
        mini = np.min(i[:,1])
        average = np.average(i[:,1])
        filewriter.writerow([maxi, mini, average])
        for row in i:
            filewriter.writerow(row)

#PLOT

fig = plt.figure(figsize=(12, 8))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plot_data = np.c_[static_columns, performed_data]

ax.scatter(plot_data[:,0], plot_data[:,1], plot_data[:,2], c=labels.astype(np.float))
for i in range(clusters_count):
    ax.scatter(np.zeros(clusters_count), np.zeros(clusters_count), centroids[i], s=100, c="red", marker="o")

# plt.show()

