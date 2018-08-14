import numpy as np
import pandas as pd
import logging
from data_loader import data_loader
from clustering_algorithms import clustering_algorithms
from data_saver import data_saver
import matplotlib.pyplot as plt

logging.basicConfig(filename='../results/log_file.log',level=logging.DEBUG)

##### File names directory
file_name = '../data/Brisbane_CityBike.json'


###########################################################################
# Hyperparameters for clustering algorithms and features names in the dataset
###########################################################################
seed = 111
features_names = ["longitude","latitude"]
algo_clustering = 'k-means'

###########################################################################
# Preparing data for clustering
###########################################################################
training_data = data_loader(file_name,features_names)
logging.info('Data has been prepared')

###########################################################################
# Training the model and get the clusters for different values of K
###########################################################################
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    results,model = clustering_algorithms(algo_clustering,training_data,k,random_state=seed)
    Sum_of_squared_distances.append(model.inertia_)
logging.info('Training has been done')

###########################################################################
# Save a plot of elbrow
###########################################################################


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k - nb of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.grid(True)
plt.savefig("../results/elbrow.png")
