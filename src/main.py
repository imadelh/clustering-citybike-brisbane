import numpy as np
import pandas as pd
import logging
from data_loader import data_loader
from clustering_algorithms import clustering_algorithms
from data_saver import data_saver
logging.basicConfig(filename='../results/log_file.log',level=logging.DEBUG)

##### File names directory
file_name = '../data/Brisbane_CityBike.json'

###########################################################################
# Hyperparameters for clustering algorithms and features names in the dataset
###########################################################################
nb_clusters = 5
seed = 111
features_names = ["longitude","latitude"]
#algo_clustering = 'k-means'
#algo_clustering = 'agglomerative'
algo_clustering = 'spectral'


###########################################################################
# Preparing data for clustering
###########################################################################
training_data = data_loader(file_name,features_names)
logging.info('Data has been prepared')

###########################################################################
# Training the model and get the clusters
###########################################################################
results,model = clustering_algorithms(algo_clustering,training_data,nb_clusters,random_state=seed,affinity = 'nearest_neighbors')
logging.info('Training has been done')


###########################################################################
# Saving results
# csv file containing data and a new column : clusters
# a PNG of initial data and new clusters
###########################################################################

features_names.append("clusters")
data_clusters = pd.DataFrame(results,columns=features_names)
data_saver(file_name,data_clusters,algo_clustering)
logging.info('Results saved')
