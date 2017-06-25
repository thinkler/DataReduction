from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import csv

df = pd.read_csv('fixed.csv')
factors_count = 4

X = df.ix[:,:].values

X_std = StandardScaler().fit_transform(X)
mean_vec = np.mean(X_std, axis=0)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

arr = []
for i in range(factors_count):
    arr.append(eig_pairs[i][1].reshape(len(eig_vals),1))

matrix_w = np.hstack(arr)

result = X_std.dot(matrix_w)

with open('output.csv', 'wb') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(["Correlation"])
    for row in cov_mat:
        filewriter.writerow(row)
    filewriter.writerow(["Eigenvectors"])
    for row in matrix_w:
        filewriter.writerow(row)
    filewriter.writerow(['New scores'])
    for row in result:
        filewriter.writerow(row)

plt.scatter(result[:, 0],
            result[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.tight_layout()
plt.show()
