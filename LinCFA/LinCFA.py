import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.decomposition import PCA
import argparse
from random import randrange
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import utils
from sklearn.preprocessing import KernelCenterer, scale
from sklearn.metrics.pairwise import pairwise_kernels
from scipy import linalg
from scipy.sparse.linalg import eigsh as ssl_eigsh
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

class LinCFA():
    """
        Class which takes as input a dataframe (or path) and the column name of the target
        The method compute_clusters prints and returns the list of aggregations with LinCFA
    """
    def __init__(self, df, target_name, eps, neigh):
        if type(df)==str:
            pd.read_csv(df)
        else: self.df = df.copy(deep=True)
        self.target_name = target_name
        self.clusters = []
        self.neigh = neigh
        
    def print_header(self):
        print("Dataset: \n{}".format(self.df))

    def compute_corr(self, column1, column2):
        return pearsonr(column1,column2)[0]
    
    def prepare_data(self, x1, x2, l):
        x1 = preprocessing.scale(x1, with_mean=True)
        x2 = preprocessing.scale(x2, with_mean=True)
        x = np.concatenate((x1.reshape(-1,1),x2.reshape(-1,1)),axis=1)
        x_aggr = ((x1*l+x2)/(l+1)).reshape(-1,1) 
        x_aggr = preprocessing.scale(x_aggr, with_mean=True)
        y = self.df[self.target_name].values
        y = preprocessing.scale(y, with_mean=True, with_std=True) 
        return x_aggr, x, y

    def compute_empirical_bound(self, column1_list, column2):

        x_aggr,x,y = self.prepare_data(self.df[column1_list].mean(axis=1).values,self.df[column2].values, len(column1_list))
        
        regr = LinearRegression()
        regr.fit(x,y)
        w1 = regr.coef_[0]
        w2 = regr.coef_[1]
        preds = regr.predict(x)
        residuals = y - preds
        
        n=len(x_aggr) 
        
        s_squared = np.dot(residuals.reshape(1,n),residuals)/(n-3)
        bound = 1 - (2*s_squared/((n-1)*(w1-w2)**2))
        corr = self.compute_corr(x[:,0].reshape(n),x[:,1].reshape(n))

        return bound, corr

    def find_neighbors(self, actual_clust, cols): 
        neighs = []
        for datum in actual_clust:
            x = float(datum.split('_')[1])
            y = float(datum.split('_')[2])
            for c in cols:
                cx = float(c.split('_')[1])
                cy = float(c.split('_')[2])
                if ((abs(x-cx)<0.1) & (abs(y-cy)<0.1)): neighs.append(c)
        return neighs
    
    def find_aggregation(self, actual_clust, cols):
        if self.neigh==1: neigh_names = self.find_neighbors(actual_clust, cols)
        else: neigh_names=cols
        for i in neigh_names:
            bound, corr = self.compute_empirical_bound(actual_clust, i)
            #print(r1,r2)
            if (corr>=bound): return i
        return ''
    
    def compute_clusters(self):
        output = []
        cols = self.df.loc[:, self.df.columns != self.target_name].columns # all the columns of the DF not yet assigned to a cluster
        actual_cluster = []

        while(len(cols)>0):

            if (actual_cluster == []):
                actual_col = cols[0] # take the first feature
                actual_cluster.append(actual_col) # append that feature to the actual cluster
                cols = cols[cols.values!=actual_col] # remove actual column from the ones not assigned yet

            col_to_aggr = self.find_aggregation(actual_cluster, cols)
            if col_to_aggr != '':
                actual_cluster.append(col_to_aggr)
                cols = cols[cols.values!=col_to_aggr]
            else:
                output.append(actual_cluster)
                actual_cluster = []
        if (len(actual_cluster)>0): output.append(actual_cluster)
        return output
    