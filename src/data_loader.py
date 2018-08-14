import pandas as pd

def data_loader(file_name,features_names):
	'''
	Inputs :
		- 'file_name' : A json file containing data
		- features_names : a list of features name in the json file
		 (Features here are supposed to be numerical, in case of categorical features (text) changes need to be made in this script to transform them to categorical features)
	outputs :

		- NP array of size (NxP) : P number of features and N number of observations
	'''

	data = pd.read_json(file_name)
	return data[features_names].values
