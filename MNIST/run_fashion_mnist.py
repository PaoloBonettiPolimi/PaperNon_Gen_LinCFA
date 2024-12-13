import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
from sklearn.metrics import r2_score
from joblib import Parallel, delayed

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.utils import resample
from keras.datasets import mnist,fashion_mnist
from sklearn.svm import SVR,SVC,LinearSVC
import pickle
import itertools
from sklearn.manifold import Isomap
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import LocallyLinearEmbedding as LLE
import umap
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA

import keras
from keras import layers
from keras.callbacks import EarlyStopping

#sys.path.append("/Users/paolo/Documents/methods/CMI_FS")
#from feature_selection import forwardFeatureSelection

# sys.path.append("../LinCFA")
# from LinCFA import LinCFA
# 
# sys.path.append("../NonLinCFA")
# from NonLinCFA import NonLinCFA
# 
# sys.path.append("../GenLinCFA")
# from GenLinCFA import GenLinCFA
# 
# sys.path.append("../droughts")
# from aux import prepare_target,prepare_features,compare_methods

#from aux import standardize,unfold_dataset,compute_r2,prepare_target,prepare_features,aggregate_unfolded_data,aggregate_unfolded_data_onlyTrain,FS_with_linearWrapper,compare_methods, compute_r2

# sys.path.append("/work/bk1318/b382633/PaperNon_Gen_LinCFA/gt_pca/code")

# from pc_layer import add_principal_component, full_embedding
# from tools_rotation import affine_transform, get_rotation_matrix
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import utils
from sklearn.preprocessing import KernelCenterer, scale
from sklearn.metrics.pairwise import pairwise_kernels
from scipy import linalg
from scipy.sparse.linalg import eigsh as ssl_eigsh
from scipy.io import arff
from sklearn.impute import SimpleImputer

class spca(BaseEstimator, TransformerMixin):
    
    def __init__(self, num_components, kernel="linear", eigen_solver='auto', 
                 max_iterations=None, gamma=0, degree=3, coef0=1, alpha=1.0, 
                 tolerance=0, fit_inverse_transform=False):
        
        self._num_components = num_components
        self._gamma = gamma
        self._tolerance = tolerance
        self._fit_inverse_transform = fit_inverse_transform
        self._max_iterations = max_iterations
        self._degree = degree
        self._kernel = kernel
        self._eigen_solver = eigen_solver
        self._coef0 = coef0
        self._centerer = KernelCenterer()
        self._alpha = alpha
        self._alphas = []
        self._lambdas = []
        
        
    def _get_kernel(self, X, Y=None):
        # Returns a kernel matrix K such that K_{i, j} is the kernel between the ith and jth vectors 
        # of the given matrix X, if Y is None. 
        
        # If Y is not None, then K_{i, j} is the kernel between the ith array from X and the jth array from Y.
        
        # valid kernels are 'linear, rbf, poly, sigmoid, precomputed'
        
        args = {"gamma": self._gamma, "degree": self._degree, "coef0": self._coef0}
        
        return pairwise_kernels(X, Y, metric=self._kernel, n_jobs=-1, filter_params=True, **args)
    
    
    
    def _fit(self, X, Y):
        
        # calculate kernel matrix of the labels Y and centre it and call it K (=H.L.H)
        K = self._centerer.fit_transform(self._get_kernel(Y))
        
        # deciding on the number of components to use
        if self._num_components is not None:
            num_components = min(K.shape[0], self._num_components)
        else:
            num_components = self.K.shape[0]
        
        # Scale X
        # scaled_X = scale(X)
        
        # calculate the eigen values and eigen vectors for X^T.K.X
        Q = (X.T).dot(K).dot(X)
        
        # If n_components is much less than the number of training samples, 
        # arpack may be more efficient than the dense eigensolver.
        if (self._eigen_solver=='auto'):
            if (Q.shape[0]/num_components) > 20:
                eigen_solver = 'arpack'
            else:
                eigen_solver = 'dense'
        else:
            eigen_solver = self._eigen_solver
        
        if eigen_solver == 'dense':
            # Return the eigenvalues (in ascending order) and eigenvectors of a Hermitian or symmetric matrix.
            self._lambdas, self._alphas = linalg.eigh(Q, eigvals=(Q.shape[0] - num_components, Q.shape[0] - 1))
            # argument eigvals = Indexes of the smallest and largest (in ascending order) eigenvalues
        
        elif eigen_solver == 'arpack':
            # deprecated :: self._lambdas, self._alphas = utils.arpack.eigsh(A=Q, num_components, which="LA", tol=self._tolerance)
            self._lambdas, self._alphas = ssl_eigsh(A=Q, k=num_components, which="LA", tol=self._tolerance)
            
        indices = self._lambdas.argsort()[::-1]
        
        self._lambdas = self._lambdas[indices]
        self._lambdas = self._lambdas[self._lambdas > 0]  # selecting values only for non zero eigen values
        
        self._alphas = self._alphas[:, indices]
        #return self._alphas
        self._alphas = self._alphas[:, self._lambdas > 0]  # selecting values only for non zero eigen values
        
        self.X_fit = X

        
    def _transform(self):
        return self.X_fit.dot(self._alphas)
        
        
    def transform(self, X):
        return X.dot(self._alphas)
        
        
    def fit(self, X, Y):
        self._fit(X,Y)
        return
        
        
    def fit_and_transform(self, X, Y):
        self.fit(X, Y)
        return self._transform()
    
def compute_CI(list,n):
    print(f'{np.mean(list)} +- {1.96*np.std(list)/np.sqrt(n)}')

def run_autoencoder(df_trainVal_withTar,df_test_withTar,target_df_trainVal,target_df_test,curr_seed,kernel=None):
    
    df_test = df_test_withTar.iloc[:,:-1]
    best_score = 0
    best_score_svr = 0
    best_num = 0
    
    for i in [2,4,8,16,32,64,128,256]:
        
        curr_df_trainVal_withTar = resample(df_trainVal_withTar, random_state=curr_seed)
        curr_df_trainVal = curr_df_trainVal_withTar.iloc[:,:-1]
        
        input_tabular = keras.Input(shape=(curr_df_trainVal.shape[1]))
        encoded = layers.Dense(i*4, activation='relu')(input_tabular)
        encoded = layers.Dense(i*2, activation='relu')(encoded)
        encoded = layers.Dense(i, activation='relu')(encoded)
        
        decoded = layers.Dense(i*2, activation='relu')(encoded)
        decoded = layers.Dense(i*4, activation='relu')(decoded)
        decoded = layers.Dense(curr_df_trainVal.shape[1], activation='linear')(decoded)
        
        autoencoder = keras.Model(input_tabular, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=20, 
                verbose=1, mode='auto', restore_best_weights=True)
        print("summary done",flush=True)
        autoencoder.fit(curr_df_trainVal, curr_df_trainVal,
                epochs=1000,
                shuffle=True,
                verbose=0,
                callbacks=[monitor],
                validation_data=(df_test, df_test))
        
        encoder = keras.Model(inputs=input_tabular, outputs=encoded)
        
        trainVal_reduced = encoder.predict(curr_df_trainVal)
        
        #mod = LinearRegression().fit(trainVal_reduced, curr_df_trainVal_withTar.mean_std)
        mod = LinearRegression().fit(trainVal_reduced, curr_df_trainVal_withTar.mean_std)
        mod_svr = SVR().fit(trainVal_reduced, curr_df_trainVal_withTar.mean_std)

        test_reduced = encoder.predict(df_test)
        print(test_reduced.shape, flush=True)
        
        actual_score = mod.score(test_reduced, df_test_withTar.mean_std)
        actual_score_svr = mod_svr.score(test_reduced, df_test_withTar.mean_std)

        #actual_score_class = mod_class.score(test_reduced, np.sign(df_test_withTar.mean_std))
        print(actual_score)
        print(actual_score_svr)
        
        #print(actual_score_class)
        if actual_score> best_score:
            best_score = actual_score
            best_num = test_reduced.shape[1]
            
        if actual_score_svr> best_score_svr:
            best_score_svr = actual_score_svr
        #if actual_score_class> best_score_class:
        #    best_score_class = actual_score_class
        #    best_num_class = test_reduced.shape[1]
    
    print([curr_seed, kernel, best_score, best_score_svr, best_num],flush=True)
    
    return [curr_seed, kernel, best_score, best_score_svr, best_num]


def run_dim_red(df_trainVal_withTar,df_test_withTar,target_df_trainVal,target_df_test,curr_seed,kernel=None):

    df_test = df_test_withTar.iloc[:,:-1]
    best_score = 0
    best_score_svr = 0
    best_num = 0
    for i in [1,2,3,4,5,7,10,12,15,20,25,30,35,40,45,50,75,100,150,200]:
        try:
            curr_df_trainVal_withTar = resample(df_trainVal_withTar, random_state=curr_seed)
            curr_df_trainVal = curr_df_trainVal_withTar.iloc[:,:-1]
            dimRedMethod = KernelPCA(n_components=i, kernel=kernel)
            # LDA(n_components=i)
            # LLE(n_components=i,n_neighbors=10)
            # KernelPCA(n_components=i, kernel=kernel)
            #spca(num_components=i, kernel=kernel, degree=3, gamma=None, coef0=1)
            #trainVal_reduced = pd.DataFrame(dimRedMethod.fit_transform(curr_df_trainVal,curr_df_trainVal_withTar.mean_std))

            trainVal_reduced = pd.DataFrame(dimRedMethod.fit_transform(curr_df_trainVal.values,curr_df_trainVal_withTar.mean_std.values.reshape(-1, 1)))
            test_reduced = pd.DataFrame(dimRedMethod.transform(df_test))
            print(test_reduced.shape, flush=True)
            mod = LogisticRegression().fit(trainVal_reduced, curr_df_trainVal_withTar.mean_std)
            mod_svr = LinearSVC().fit(trainVal_reduced, curr_df_trainVal_withTar.mean_std)
            actual_score = mod.score(test_reduced, df_test_withTar.mean_std)
            actual_score_svr = mod_svr.score(test_reduced, df_test_withTar.mean_std)
            print(actual_score,flush=True)
            if actual_score> best_score:
                best_score=actual_score
                best_num = test_reduced.shape
            if actual_score_svr> best_score_svr:
                best_score_svr=actual_score_svr
        except:
            print("error")
            pass
    print([curr_seed, kernel, best_score, best_score_svr, best_num],flush=True)
    
    return [curr_seed, kernel, best_score, best_score_svr, best_num]

def run_dim_red_parallel(df_trainVal_withTar,df_test_withTar,target_df_trainVal,target_df_test):

    #result = Parallel(n_jobs=-1)(delayed(run_autoencoder)(df_trainVal_withTar,df_test_withTar,target_df_trainVal,target_df_test,curr_seed) for curr_seed in [0,1,2,3,4])

    #result = Parallel(n_jobs=-1)(delayed(run_dim_red)(df_trainVal_withTar,df_test_withTar,target_df_trainVal,target_df_test,curr_seed) for curr_seed in [0,1,2,3,4])
    result = Parallel(n_jobs=-1)(delayed(run_dim_red)(df_trainVal_withTar,df_test_withTar,target_df_trainVal,target_df_test,curr_seed,kernel) for curr_seed,kernel in list(itertools.product([0,1,2,3,4],['linear','poly','sigmoid'])))

    return result


### main run
if __name__ == "__main__":

    n_repetitions = 5
    algo = "regression"

    ##################### data ########################

    if algo=="regression":
        (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
        
        df_train = pd.DataFrame(train_X.reshape(train_X.shape[0],-1))
        df_train["mean_std"] = train_y.reshape(train_y.shape[0],-1)
        df_train = df_train[(df_train.mean_std==7) | (df_train.mean_std==1)].reset_index(drop=True)
        
        df_test = pd.DataFrame(test_X.reshape(test_X.shape[0],-1))
        df_test["mean_std"] = test_y.reshape(test_y.shape[0],-1)
        df_test = df_test[(df_test.mean_std==7) | (df_test.mean_std==1)].reset_index(drop=True)
        
        print(df_train.shape, df_test.shape, flush=True)
    
        df_trainVal_withTar = df_train.iloc[:,:-1]
        df_trainVal_withTar["mean_std"] = df_train.iloc[:,267]
        df_trainVal_withTar = df_trainVal_withTar.iloc[:,list(range(267))+list(range(268,785))]
        
        df_test_withTar = df_test.iloc[:,:-1]
        df_test_withTar["mean_std"] = df_test.iloc[:,267]
        df_test_withTar = df_test_withTar.iloc[:,list(range(267))+list(range(268,785))]
        
        df_trainVal = df_trainVal_withTar.iloc[:,:-1]
        df_test = df_test_withTar.iloc[:,:-1]
        target_df_trainVal = df_trainVal_withTar.iloc[:,-1]
        target_df_test = df_test_withTar.iloc[:,-1]
    else:

        (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
        
        df_train = pd.DataFrame(train_X.reshape(train_X.shape[0],-1))
        df_train["mean_std"] = train_y.reshape(train_y.shape[0],-1)
        df_train = df_train[(df_train.mean_std==0) | (df_train.mean_std==2) | (df_train.mean_std==4)].reset_index(drop=True)
        
        df_test = pd.DataFrame(test_X.reshape(test_X.shape[0],-1))
        df_test["mean_std"] = test_y.reshape(test_y.shape[0],-1)
        df_test = df_test[(df_test.mean_std==0) | (df_test.mean_std==2) | (df_test.mean_std==4)].reset_index(drop=True)
        
        print(df_train.shape, df_test.shape, flush=True)
        
        df_trainVal_withTar = df_train
        df_test_withTar = df_test
        df_trainVal = df_train.iloc[:,:-1]
        df_test = df_test.iloc[:,:-1]
        target_df_trainVal = df_train.iloc[:,-1]
        target_df_test = df_test.iloc[:,-1]

    ##################### experiment ########################
    res = run_dim_red_parallel(df_trainVal_withTar,df_test_withTar,target_df_trainVal,target_df_test)
    
    ##################### save the results ########################
    
    with open('fashionMnist_kernelPCA.pkl', 'wb') as f:  
         pickle.dump(res, f)
