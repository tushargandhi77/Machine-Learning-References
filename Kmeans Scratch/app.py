from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans import KMeans

centroids = [(-5,-5),(5,5),(-2.5,2.5)]
cluster_std = [1,1,1]


X,y = make_blobs(n_samples=100,cluster_std=cluster_std,centers=centroids,n_features=2,random_state=2)


km = KMeans(n_clusters=3,max_iter=1000)
y_means = km.fit_predict(X)

plt.scatter(X[y_means == 0,0], X[y_means == 0,1], c='red')
plt.scatter(X[y_means == 1,0], X[y_means == 1,1], c='blue')
plt.scatter(X[y_means == 2,0], X[y_means == 2,1], c='green')

# plt.scatter(X[:,0],X[:,1])
plt.show()