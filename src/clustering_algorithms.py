import numpy as np
import inspect
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering

algos = {'k-means': KMeans,'agglomerative' : AgglomerativeClustering, 'spectral': SpectralClustering}

def clustering_algorithms(algo_type,data,nb_clusters,**kwargs):
    '''
    Inputs :
        - algo_type : name of clustering algo
        - data : np array of shape : (NB_OBSERVATIONS x FEATURES)
        - nb_clusters : number of clusters
        - **kwargs : additional parameters that are specefic to the algorithm (see documentation of algorihtms for the names of parameters :
            http://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
            http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
            )

    Outputs :

        - np array like : [data,clusters]
        - trained model
    '''
    model_exists = algo_type in algos
    if(model_exists == False):
        print("Algorithm",algo_type," is not defined","Please import it and add it to 'algos' in clustering_algorithms.py")
        exit()

    else:

        # To keep the program more flexible, if the user gives a kwargs that is not used in some algorithm, we will delete that args and inform the user
        algorithm_args = list(inspect.signature(algos[algo_type]).parameters.keys())
        for key in list(kwargs):
            if(key not in algorithm_args):
                print("##############")
                print(key,'is not in the algorithm (',algo_type,') arguments')
                print('Therefore it was omitted')
                print("##############")
                del kwargs[key]

        # Train the model
        clf = algos[algo_type](n_clusters = nb_clusters,**kwargs)
        Y = clf.fit_predict(data)
        Y = np.expand_dims(Y, axis=1)

    # return the results
    return np.hstack((data,Y)), clf
