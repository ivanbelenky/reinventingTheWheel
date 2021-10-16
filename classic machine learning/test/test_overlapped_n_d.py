from .. import utils as u
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1)
nd = u.generate_overlapped_normal_distribution(dimension=2, n_points=1000)

for n in nd:
    ax.scatter(n[:,0],n[:,1])

plt.show()