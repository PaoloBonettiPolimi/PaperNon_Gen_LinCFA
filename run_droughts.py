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
    curr_df_trainVal_withTar = pd.concat((curr_df_trainVal,target_df_trainVal['mean_std']), axis=1)
        
    output = NonLinCFA.NonLinCFA(curr_df_trainVal_withTar,'mean_std', eps, -5 , 0).compute_clusters()
        
    aggregate_trainVal = pd.DataFrame()
    aggregate_test = pd.DataFrame()
    for i in range(len(output)):
        aggregate_trainVal[str(i)] = curr_df_trainVal_withTar[output[i]].mean(axis=1)
        aggregate_trainVal = aggregate_trainVal.copy()
        aggregate_test[str(i)] = curr_df_test[output[i]].mean(axis=1)
        aggregate_test = aggregate_test.copy()
    print(f'Number of aggregated features: {len(output)}\n')
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

    target_df_train,target_df_val,target_df_test,target_df_trainVal = prepare_target('',max_train='2010-01-01', max_val='2015-01-01', max_test='2020-01-01', path='../PaperNon_Gen_LinCFA/droughts/Emiliani1.csv')
    
    ### Load and standardize features
    variables_list = ['cyclostationary_mean_tg', 
                    'cyclostationary_mean_tg_1w',
                    'cyclostationary_mean_tg_4w', 
                    'cyclostationary_mean_tg_8w',
                    'cyclostationary_mean_tg_12w', 
                    'cyclostationary_mean_tg_16w',
                    'cyclostationary_mean_tg_24w',
                    'cyclostationary_mean_rr', 
                    'cyclostationary_mean_rr_1w',
                    'cyclostationary_mean_rr_4w', 
                    'cyclostationary_mean_rr_8w',
                    'cyclostationary_mean_rr_12w', 
                    'cyclostationary_mean_rr_16w',
                    'cyclostationary_mean_rr_24w'
                    ]

    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()
    df_trainVal = pd.DataFrame()

    for variable in variables_list:
        df_train_unfolded_std, df_val_unfolded_std, df_test_unfolded_std,df_trainVal_unfolded_std = prepare_features('../PaperNon_Gen_LinCFA/droughts/Emiliani1_aggreg.csv',variable,False,max_train='2010-01-01', max_val='2015-01-01', max_test='2020-01-01')
        df_train_unfolded_std = df_train_unfolded_std.add_prefix(variable)
        df_val_unfolded_std = df_val_unfolded_std.add_prefix(variable)
        df_test_unfolded_std = df_test_unfolded_std.add_prefix(variable)
        df_trainVal_unfolded_std = df_trainVal_unfolded_std.add_prefix(variable)
        df_train = pd.concat((df_train,df_train_unfolded_std),axis=1)
        df_val = pd.concat((df_val,df_val_unfolded_std),axis=1)
        df_test = pd.concat((df_test,df_test_unfolded_std),axis=1)
        df_trainVal = pd.concat((df_trainVal,df_trainVal_unfolded_std),axis=1)
        
    df_trainVal_withTar = pd.concat((df_trainVal,target_df_trainVal['mean_std']), axis=1)

    ##################### experiment ########################
    res = run_NonLinCFA_parallel(df_trainVal,df_test,target_df_trainVal,target_df_test,args.n_repetitions)
    
    ##################### save the results ########################
    
    with open(args.results_file, 'wb') as f:  
        pickle.dump(res, f)
