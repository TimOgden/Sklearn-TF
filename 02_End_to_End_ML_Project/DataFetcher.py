
# coding: utf-8

# In[2]:


import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# In[3]:


DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/' #Where to fetch the data
HOUSING_PATH = os.path.join('datasets','housing') #Where in the local directory to store the data
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
	csv_path = os.path.join(housing_path, 'housing.csv')
	if not os.path.isfile(csv_path):
		fetch_housing_data()
	return pd.read_csv(csv_path)

df = load_housing_data()
#print(df.describe())
#df.hist(bins=50,figsize=(20,15))
#plt.show()

def split_train_set(data, test_ratio=0.2):
	shuffled_indices = np.random.permutation(len(data)) #What does np.random.permutation() do?
	test_set_size = int(len(data) * test_ratio)
	test_indices = shuffled_indices[:test_set_size]
	train_indices = shuffled_indices[test_set_size:]
	return data.iloc[train_indices], data.iloc[test_indices]

