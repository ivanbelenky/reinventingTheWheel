import utils as u
import numpy as np
import matplotlib.pyplot as plt
from k_nearest_neighbors import maj_knn

plt.style.use("dark_background")

NUMCLASSES = 5
NPOINTS = 5
DIMENSION = 2
K = 7

_,nds = u.generate_overlapped_normal_distribution(
    p=NUMCLASSES,
    dimension=DIMENSION, 
    n_points=NPOINTS    
    )

data = np.vstack(nds)
labels = np.array([[j for _ in range(NPOINTS)] for j in range(len(nds))]).reshape(-1)


x_min, x_max = data[:,0].min() -0.1, data[:,0].max() + 0.1
y_min, y_max = data[:,1].min() -0.1 , data[:,1].max() + 0.1

xx, yy = np.meshgrid(np.arange(x_min,x_max,step=(x_max-x_min)/10),
    np.arange(y_min,y_max,step=(y_max-y_min)/10))

targets = np.c_[xx.ravel(), yy.ravel()]

z = np.array([maj_knn(t, K, data, labels, np.linalg.norm) for t in targets])
z = z.reshape(xx.shape)


fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.contourf(xx,yy,z,alpha=0.4)
for nd in nds:
    ax.scatter(nd[:,0],nd[:,1],s=20,edgecolors='gray')

plt.show()
