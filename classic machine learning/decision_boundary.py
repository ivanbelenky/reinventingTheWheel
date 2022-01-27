import utils as u
import numpy as np
import matplotlib.pyplot as plt
from k_nearest_neighbors import maj_knn

from matplotlib import colors

plt.style.use("dark_background")

NUMCLASSES = 3
NPOINTS = 500
DIMENSION = 2
K = 3

_,nds = u.generate_overlapped_normal_distribution(
    p=NUMCLASSES,
    dimension=DIMENSION, 
    n_points=NPOINTS    
    )

data = np.vstack(nds)
labels = np.array([[j for _ in range(NPOINTS)] for j in range(len(nds))]).reshape(-1)


x_min, x_max = data[:,0].min() -0.5, data[:,0].max() + 0.5
y_min, y_max = data[:,1].min() -0.5 , data[:,1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min,x_max,step=(x_max-x_min)/100),
    np.arange(y_min,y_max,step=(y_max-y_min)/100))

targets = np.c_[xx.ravel(), yy.ravel()]

z = np.array([maj_knn(t, K, data, labels, np.linalg.norm) for t in targets])
z = z.reshape(xx.shape)

cmap = colors.ListedColormap(['r','b','g'])
fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.set_title(f"Decision Boundary for K={K}")
ax.contourf(xx,yy,z,alpha=0.6,cmap = cmap)
for nd,c in zip(nds,['r','b','g']):
    ax.scatter(nd[:,0],nd[:,1],s=20,edgecolors='gray',color=c)

ax.set_ylabel("x2")
ax.set_xlabel("x1")

plt.show()
