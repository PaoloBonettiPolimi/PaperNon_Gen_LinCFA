import sys
import pandas as pd
sys.path.append("../PaperNon_Gen_LinCFA/")
from NonLinCFA.NonLinCFA import NonLinCFA
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import r2_score,accuracy_score
import matplotlib.pyplot as plt
from NonLinCFA.NonLinCFA_anyFunction import NonLinCFA_anyFunction
from LinCFA.LinCFA import LinCFA
from GenLinCFA.GenLinCFA import GenLinCFA
from GenLinCFA.GenLinCFA_anyFunction import GenLinCFA_anyFunction
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import random
from sklearn import preprocessing
import pickle
import argparse
from joblib import Parallel, delayed

def compute_accuracy(x_train, y_train, x_val, y_val):
    regr = LogisticRegression().fit(x_train,y_train)
    y_pred = regr.predict(x_val)
    return accuracy_score(y_val, y_pred)

def compute_wrapper(x_trainVal, y_trainVal,n_features):
    sfs = SFS(LogisticRegression(),
           k_features=n_features,
           forward=True,
           floating=False,
           scoring = 'accuracy',
           cv = 5,
	   n_jobs=5)
    
    sfs.fit(x_trainVal, y_trainVal)
    return pd.DataFrame(sfs.subsets_).transpose()

def squared_aggregation(x):
    x_sq = x**2
    return x_sq.sum(axis=1)

def mean_aggregation(x):
    return x.mean(axis=1)

def run_GenLinCFA(x,eps,passed_fun):
    output = GenLinCFA_anyFunction(x.iloc[:2000,:],'target', eps1=eps, n_val=5 , neigh=0, eps2=1/2, customFunction=passed_fun).compute_clusters()

    aggregate_x = pd.DataFrame()
    for i in range(len(output)):
        aggregate_x[i] = passed_fun(x[output[i]])
        
    actual_score = compute_accuracy(aggregate_x.iloc[:2000,:], x.loc[:,'target'].iloc[:2000], aggregate_x.iloc[2000:,:], x.loc[:,'target'].iloc[2000:])

    return [eps,len(output),actual_score]

def quadratic_experiment_noResampling(n_reps=10, n_variables=100, noise=10, hyper=[0.5]):
    NonLinCFA_score = [] 
    list_of_length = []
    wrapper_score = []
    LinCFA_score = []
    list_of_length_lin = []
    GenLinCFA_score = []
    list_of_length_gen = []
    NonLinCFA_score_sq = [] 
    GenLinCFA_score_sq = []
    wrapper_score_sq = []

    x_all,y_all,coeffs = generate_dataset_squared(n_data=3000*10, noise=noise, p1=0.3, p2=0.7, n_variables=n_variables, coeffs=[0])
    mean_target = np.mean(y_all)
    for trials in range(n_reps):
        x = pd.DataFrame(x_all[3000*trials:3000*(trials+1)])
        x_squared = x**2
        x['target'] = y_all[3000*trials:3000*(trials+1)]
        x['target'] = x.apply(lambda x: 1 if x.target>=mean_target else 0,axis=1)
        x_squared['target'] = y_all[3000*trials:3000*(trials+1)]
        x_squared['target'] = x_squared.apply(lambda x: 1 if x.target>=mean_target else 0,axis=1)

        results = Parallel(n_jobs=10)(delayed(run_GenLinCFA)(x_squared,eps,mean_aggregation) for eps in hyper)
        print(results)
        GenLinCFA_score.append(results)

        results = Parallel(n_jobs=10)(delayed(run_GenLinCFA)(x,eps,squared_aggregation) for eps in hyper)
        print(results)
        GenLinCFA_score_sq.append(results)
        
        #wrapper
        res = compute_wrapper(x_squared.iloc[:2000,:-1], x_squared.iloc[:2000,-1], 50)
        for i in range(50):
            x_wrapper_red = x_squared[list(res.loc[i+1,'feature_names'])]
            wrapper_score.append(compute_accuracy(x_wrapper_red.iloc[:2000],x_squared.loc[:,'target'].iloc[:2000],x_wrapper_red.iloc[2000:],x_squared.loc[:,'target'].iloc[2000:]))
            print(i+1,wrapper_score[-1])
        
    return GenLinCFA_score,GenLinCFA_score_sq,wrapper_score

### generate a dataset of n samples and standardize the variables x1,x2,x3
def generate_dataset_squared(n_data=3000, noise=1, p1=0.5, p2=0.5, n_variables=10, coeffs=[0]):
    x = np.zeros((n_data,n_variables))
    
    rng = np.random.default_rng(seed=0)
    random.seed(30)

    x[:,0] = rng.uniform(0,1,n_data)
    if coeffs[0]==0:
        coeffs = rng.uniform(0,1,size=n_variables)
    
    for i in range(n_variables-1):
        j = random.randrange(i+1)
        x[:,i+1] = p1*rng.uniform(0,1,n_data) + p2*x[:,j]
    
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)
    
    delta = rng.normal(0, noise, size=(n_data,1))
    
    y = np.dot(x**2,coeffs).reshape(n_data,1) + delta.reshape(n_data,1)
    return x,y,coeffs

### main run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_variables", default=100, type=int)
    parser.add_argument("--noise", default=10, type=float)
    parser.add_argument("--n_repetitions", default=500, type=int)
    parser.add_argument("--p1", default=0.3, type=float)
    parser.add_argument("--p2", default=0.7, type=float)
    parser.add_argument("--results_file", default='res.pkl')
    parser.add_argument("--hyperparameter", nargs="*", default=[0.5], type=float)

    args = parser.parse_args()
    print(args)

    ##################### experiment ########################

    GenLinCFA_score,GenLinCFA_score_sq,wrapper_score = quadratic_experiment_noResampling(n_reps=args.n_repetitions, n_variables=args.n_variables, noise=args.noise, hyper = args.hyperparameter)
    
    ##################### save the results ########################
    
    with open(args.results_file, 'wb') as f:  
        pickle.dump([GenLinCFA_score,GenLinCFA_score_sq,wrapper_score], f)
