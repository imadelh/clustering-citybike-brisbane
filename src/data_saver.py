import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def data_saver(file_name,data_clustered,name_algo):
    '''
    Inputs :
        - 'file_name' : A json file containing data
        - data_clustered : a pandas file containing features and an additional column : 'clusters'
    outputs :
        - Pandas data frame (saved in csv) containing the initial data (with all the columns) and 'clusters'
    '''

    data = pd.read_json(file_name)
    data = data.merge(data_clustered,how='inner')
    data.to_csv('../results/clustered_data.csv',sep = ";")
    plt.figure(figsize=(16,6.5))
    plt.subplot(1,2,1)
    plt.plot(data['longitude'], data['latitude'],'.')
    plt.title(" Map of initial stations ")
    plt.xlabel('longitude ')
    plt.ylabel('latitude')
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.scatter(data['longitude'], data['latitude'], c=data['clusters'])
    plt.title(" Map of clustered stations (each color represents a cluster) ")
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.grid(True)

    plt.savefig("../results/plot_clusters_"+name_algo+".png")
