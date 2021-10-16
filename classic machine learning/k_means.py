import numpy as np
import utils as u
import matplotlib.pyplot as plt

plt.style.use('dark_background')

MAX_ITER = 100000
TOLERANCE = 1E-5

def k_means_clustering(data, k):
    """"
    Parameters
    ----------
    data: ~numpy.ndarray 
        Each data point is a (n,) shaped ndarray. There are N points.
        Therefore data is (n,N) shaped ndarray. 
    k : int
        number of means to estimate

    Returns
    -------
    (new_means, mean_evolution) : tuple
        new means: means found after MAXITER or first iteration going below Tolerance.
        mean_evolution: every mean until break condition is achieved.

    """
    d = data.shape[1]
    initial_means = u.random(size=(k,d))
    iter_count = 0

    mean_evolution = []
    old_means = initial_means
    while(iter_count < MAX_ITER):
        new_means  = get_new_means(data,old_means)
        if np.linalg.norm(new_means-old_means) < TOLERANCE:
            break
        old_means = new_means
        mean_evolution.append(new_means)
    
    return new_means, mean_evolution

def get_new_means(data, means):
    distance_from_means = [np.linalg.norm(data-m,axis=1) for m in means]
    
    distances = (dm for dm in distance_from_means)
    distances_hstacked = np.vstack(distances).T
    
    mask = np.argmin(distances_hstacked, axis=1)

    new_clusters = [data[mask == cluster] for cluster in range(means.shape[0])]
    
    new_means = []
    for nc,mean in zip(new_clusters, means):
        new_mean = np.mean(nc, axis=0)

        if np.isnan(new_mean).any():
            new_means.append(mean)
        else:
            new_means.append(new_mean)
    
    new_means = np.array(new_means)

    return new_means


def main():
    
    initial_means,nds = u.generate_overlapped_normal_distribution(p=6, dimension=2, n_points=10000)
    
    data = [nd for nd in nds]
    data = np.vstack(data)

    new_means, mean_evolution = k_means_clustering(data,6)
    
    fig,ax = plt.subplots(2,1,figsize=(10,20))
    for nd in nds:
        ax[0].scatter(nd[:,0],nd[:,1],s=0.1)
    
    for mean,new_mean in zip(initial_means,new_means):
        ax[0].scatter(mean[0],mean[1],color='w',s=100,marker='+')
        ax[0].scatter(new_mean[0], new_mean[1],color='w',s=100,marker='*')

    mean_evolution_length = [ np.linalg.norm(mean_evolution[i]-mean_evolution[i-1]) 
    for i in range(1,len(mean_evolution))]

    generations = [i for i in range(1, len(mean_evolution))]
    
    ax[1].plot(generations, mean_evolution_length)

    plt.show()


if __name__ == "__main__":
    main()








