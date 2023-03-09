import argparse
import glob
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from pathlib import Path
from sklearn import preprocessing
from sklearn.utils import check_random_state
from scipy.stats import pearsonr
import multiprocessing
from joblib import Parallel, delayed
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from scipy.stats import pearsonr
import csv 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
import pickle

class GenLinCFA_anyFunction():
    """
        Class which takes as input a dataframe (or path), the column name of the target, the allowed error and the number of cross-validation batches
        The method compute_clusters prints and returns the list of aggregations
    """
    def __init__(self, df, target_name, eps1, n_val, neigh,  eps2, customFunction , scale=0.1):
        if type(df)==str:
            pd.read_csv(df)
        else: self.df = df.copy(deep=True)
        self.target_name = target_name
        self.eps1 = eps1
        self.eps2 = eps2
        self.n_val = n_val
        self.clusters = []
        self.scale = scale
        self.neigh = neigh
        self.fun = customFunction

    def print_header(self):
        print("Dataset: \n{}".format(self.df))

    def compute_corr(self, column1, column2):
        return pearsonr(self.df[column1],self.df[column2])[0]
    
    def prepare_data(self, x1, x2):
        x_aggr = self.fun(pd.concat((x1,x2),axis=1))
        x1 = self.fun(x1)
        x1 = preprocessing.scale(x1, with_mean=True)
        x2 = preprocessing.scale(x2, with_mean=True)
        x = np.concatenate((x1.reshape(-1,1),x2.reshape(-1,1)),axis=1)
        x_aggr = preprocessing.scale(x_aggr, with_mean=True).reshape(-1, 1)
        y = self.df[self.target_name].values
        y = preprocessing.scale(y, with_mean=True, with_std=True) 
        return x_aggr, x, y

    def find_neighbors(self, actual_clust, cols): 
        neighs = []
        for datum in actual_clust:
            x = float(datum.split('_')[1])
            y = float(datum.split('_')[2])
            for c in cols:
                cx = float(c.split('_')[1])
                cy = float(c.split('_')[2])
                if ((abs(x-cx)<self.scale) & (abs(y-cy)<self.scale)): neighs.append(c) # for droughts self.scale=0.1
        return neighs
    
    def find_aggregation(self, actual_clust, cols):
        if self.neigh==1: neigh_names = self.find_neighbors(actual_clust, cols)
        else: neigh_names=cols
        for i in neigh_names:
            cov_x1_y,cov_x2_y,cov_xmean_y,var_x1x2,var_xmean,var_y = self.compute_scores(actual_clust, i)
            #if ((abs(cov_x1_y)+abs(cov_x2_y)-abs(cov_xmean_y)<=self.eps1) & (var_x1x2-var_xmean<=self.eps2)): return i
            #print((abs(cov_x1_y)+abs(cov_x2_y)-0.5*var_x1x2))
            #print(abs(cov_x1_y)+abs(cov_x2_y)-self.eps1*0.5*var_x1x2)
            if ((abs(cov_x1_y)+abs(cov_x2_y)-self.eps2*0.5*var_xmean-self.eps1*(abs(cov_xmean_y)+self.eps2*0.5*var_x1x2))<=0): return i
        return ''

    def compute_scores(self, column1_list, column2):
        x_aggr,x,y = self.prepare_data(self.df[column1_list],self.df[column2])
        cov_x1_y = np.cov(x[:,0],y)
        cov_x2_y = np.cov(x[:,1],y)
        cov_xmean_y = np.cov(x_aggr.reshape(1,-1),y.reshape(1,-1))
        var_x1x2 = np.var(x[:,0]+x[:,1])
        var_xmean = np.var(x_aggr.reshape(1,-1))
        var_y = np.var(y.reshape(1,-1))
        return cov_x1_y[0,1],cov_x2_y[0,1],cov_xmean_y[0,1],var_x1x2,var_xmean,var_y
    
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
    