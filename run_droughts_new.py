import sys
import pandas as pd
from NonLinCFA.NonLinCFA import NonLinCFA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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

sys.path.append("../LinCFA")
from LinCFA import LinCFA

sys.path.append("../NonLinCFA")
from NonLinCFA import NonLinCFA

sys.path.append("../GenLinCFA")
from GenLinCFA import GenLinCFA

sys.path.append("../droughts")
from droughts.aux import prepare_target,prepare_features,compare_methods

def run_NonLinCFA(df_trainVal,df_test,target_df_trainVal,target_df_test,n_reps,curr_seed,eps):
    res = []
    #for eps in [0.01,0.001,0.0001,0.00001,0.000001]:
    print(f'Started: {curr_seed},{eps}')
    curr_df_trainVal = df_trainVal[np.random.default_rng(seed=curr_seed).permutation(df_trainVal.columns.values)]
    curr_df_test = df_test[np.random.default_rng(seed=curr_seed).permutation(df_test.columns.values)]
    curr_df_trainVal_withTar = pd.concat((curr_df_trainVal,target_df_trainVal), axis=1)
        
    output = NonLinCFA.NonLinCFA(curr_df_trainVal_withTar,'mean_std', eps, -5 , 0).compute_clusters()
        
    aggregate_trainVal = pd.DataFrame()
    aggregate_test = pd.DataFrame()
    for i in range(len(output)):
        aggregate_trainVal[str(i)] = curr_df_trainVal_withTar[output[i]].mean(axis=1)
        aggregate_trainVal = aggregate_trainVal.copy()
        aggregate_test[str(i)] = curr_df_test[output[i]].mean(axis=1)
        aggregate_test = aggregate_test.copy()
    print(f'Number of aggregated features: {len(output)}, with epsilon and seed {eps}{curr_seed}\n')
    r2 = compare_methods(aggregate_trainVal, aggregate_test, target_df_trainVal, target_df_test, list(aggregate_trainVal.columns))
    #res.append([eps,curr_seed,len(output),r2])
    return [eps,curr_seed,len(output),r2]

def run_NonLinCFA_parallel(df_trainVal,df_test,target_df_trainVal,target_df_test,n_reps):
    
    l = []
    for i in [0.01,0.001,0.0001,0.00001,0.000001]:
        for j in [0,1,2,3,4]:
            l.append([i,j])

    result = Parallel(n_jobs=-1)(delayed(run_NonLinCFA)(df_trainVal,df_test,target_df_trainVal,target_df_test,n_reps,curr_seed,eps) for eps,curr_seed in l)

    return result


### main run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_repetitions", default=5, type=int)
    parser.add_argument("--results_file", default='res.pkl')

    args = parser.parse_args()
    print(args)

    ##################### data ########################

    df = pd.read_csv('../PaperNon_Gen_LinCFA/droughts/droughts_extended.csv')
    df_trainVal_withTar = df.iloc[:-392,:]
    df_test_withTar = df.iloc[-392:,:]
    df_trainVal = df.iloc[:-392,:-1]
    df_test = df.iloc[-392:,:-1]
    target_df_trainVal = df.iloc[:-392,-1]
    target_df_test = df.iloc[-392:,-1]

    ##################### experiment ########################
    res = run_NonLinCFA_parallel(df_trainVal,df_test,target_df_trainVal,target_df_test,args.n_repetitions)
    
    ##################### save the results ########################
    
    with open(args.results_file, 'wb') as f:  
        pickle.dump(res, f)
