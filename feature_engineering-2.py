import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from tqdm import tqdm_notebook
import warnings
from statsmodels.tsa.stattools import kpss
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold, KFold
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


def add_rolling_features(df, window_sizes, multibatch=True):
    num_objects = df.shape[0]
    batch_size = 500*(10**3)
    num_batches = num_objects // batch_size
    
    df['batch'] = df.index // batch_size
    
    for window in tqdm_notebook(window_sizes):
        df["rolling_mean_" + str(window)] = df['signal'].rolling(window=window).mean()
        df["rolling_std_" + str(window)] = df['signal'].rolling(window=window).std()
        df["rolling_var_" + str(window)] = df['signal'].rolling(window=window).var()
        df["rolling_min_" + str(window)] = df['signal'].rolling(window=window).min()
        df["rolling_max_" + str(window)] = df['signal'].rolling(window=window).max()
        df["rolling_median_" + str(window)] = df['signal'].rolling(window=window).median()
    
        df["rolling_min_max_ratio_" + str(window)] = df["rolling_min_" + str(window)] \
                                                     / df["rolling_max_" + str(window)]
        df["rolling_min_max_diff_" + str(window)] = df["rolling_max_" + str(window)] \
                                                    - df["rolling_min_" + str(window)]
    
        a = (df['signal'] - df['rolling_min_' + str(window)]) \
            / (df['rolling_max_' + str(window)] - df['rolling_min_' + str(window)])
        df["norm_" + str(window)] = a * (np.floor(df['rolling_max_' + str(window)]) \
                                         - np.ceil(df['rolling_min_' + str(window)]))
        
    df = df.replace([np.inf, -np.inf], np.nan)
    df.fillna(0, inplace=True)
    
    return df

def add_batch_stats(df, batch_size = 50 * 10**4):
    df = df.copy()
    df['batch'] = df.index // batch_size
    df['batch_index'] = df.index - df['batch'] * batch_size
    
    df['batch_min'] = df.groupby('batch')['signal'].transform(np.min)
    df['batch_max'] = df.groupby('batch')['signal'].transform(np.max)
    df['batch_median'] = df.groupby('batch')['signal'
                                                  ].transform(np.median)
    df['batch_mode'] = df.groupby('batch')['signal'].transform(
        lambda x: sps.mode(x)[0][0])
    
    df['batch_min_abs'] = df.groupby('batch')['signal'].transform(
    lambda x: np.min(np.abs(x)))
    df['batch_max_abs'] = df.groupby('batch')['signal'].transform(
    lambda x: np.max(np.abs(x)))
    df['batch_median_abs'] = df.groupby('batch')['signal'].transform(
    lambda x: np.median(np.abs(x)))
    df['batch_mode_abs'] = df.groupby('batch')['signal'].transform(
    lambda x: sps.mode(np.abs(x))[0][0])
    
    return df

def scaling(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df)


def exp_array_smoothing(y, alpha):
    res = np.zeros(len(y))
    res[0] = y[0]
    
    for i in range(1, len(y)):
        res[i] = res[i-1] + alpha*(y[i] - res[i-1])
        
    return res


def exponential_smoothing(df, alphas):
    for alpha in alphas:
        df['exp_' + str(alpha)] = exp_array_smoothing(np.array(df['signal']), 
                                                      alpha)
        
    return df


def signal_shifts(df, shifts):
    for shift in shifts:
        df['shift_'+str(shift)] = df.signal.shift(shift)
        
    df = df.replace([np.inf, -np.inf], np.nan)
    df.fillna(0, inplace=True)
    
    return df


def batch_stats2(df, batch_sizes):
    for batch_size in batch_sizes:
        df['tmp_index'] = df.index // batch_size
        d = {}
        d[f'mean_batch{batch_size}'] = df.groupby(['tmp_index'])['signal'].mean()
        d[f'median_batch{batch_size}'] = df.groupby(['tmp_index'])['signal'].median()
        d[f'max_batch{batch_size}'] = df.groupby(['tmp_index'])['signal'].max()
        d[f'min_batch{batch_size}'] = df.groupby(['tmp_index'])['signal'].min()
        d[f'std_batch{batch_size}'] = df.groupby(['tmp_index'])['signal'].std()
        d[f'mean_abs_chg_batch{batch_size}'] = df.groupby(['tmp_index'])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))
        d[f'abs_max_batch{batch_size}'] = df.groupby(['tmp_index'])['signal'].apply(lambda x: np.max(np.abs(x)))
        d[f'abs_min_batch{batch_size}'] = df.groupby(['tmp_index'])['signal'].apply(lambda x: np.min(np.abs(x)))
        d[f'max-min_batch{batch_size}'] = d[f'max_batch{batch_size}'] - \
                                            d[f'min_batch{batch_size}']
        d[f'max/min_batch{batch_size}'] = d[f'max_batch{batch_size}'] / d[f'min_batch{batch_size}']
        d[f'abs_avg_batch{batch_size}'] = (d[f'abs_min_batch{batch_size}'] + d[f'abs_max_batch{batch_size}']) / 2
        for v in d:
            df[v] = df['tmp_index'].map(d[v].to_dict())
            
    df = df.drop(columns=['tmp_index'])
            
    return df

def add_minus_signal(df):
    for feat in [feat_ for feat_ in df.columns if feat_ not in ['time', 'signal', 'open_channels', 'batch']]:
        df[feat + '_msignal'] = df[feat] - df['signal']
        
    return df


def delete_objects_after_rolling(df, n):
    num_batches = df.shape[0] // 500000
    indices_to_delete = []
    for i in range(num_batches):
        indices_to_delete += list(range(i*500000, i*500000+n))
        
    df = df.drop(index=indices_to_delete)
    
    return df


def add_quantiles(train, test, n_bins_arr):
    for n_bins in n_bins_arr:
        binner = KBinsDiscretizer(n_bins, encode='ordinal')
        binner.fit(train.signal.values.reshape(-1, 1))
        train[f'quant_{n_bins}'] = binner.transform(train.signal.values.reshape(-1, 1)).astype('int').flatten()
        test[f'quant_{n_bins}'] = binner.transform(test.signal.values.reshape(-1, 1)).astype('int').flatten()


def add_target_encoding(train, test, n_bins_arr):
    # обычный target encoding для теста
    for n_bins in tqdm_notebook(n_bins_arr):
        train_quant_channel = train[[f'quant_{n_bins}', 'open_channels']]
        train_encoding_mean = train_quant_channel.groupby(f'quant_{n_bins}').mean()
        train_encoding_std = train_quant_channel.groupby(f'quant_{n_bins}').std()
        train_encoding_var = train_quant_channel.groupby(f'quant_{n_bins}').var()
        
        d = {}
        for q, v in zip(train_encoding_mean.index.values,
                        train_encoding_mean['open_channels'].values):
            if q not in d:
                d[q] = v
        test_values = []
        for q in test[f'quant_{n_bins}'].values:
            test_values.append(d[q])
        test[f'quant_{n_bins}_mean'] = test_values
        
        d = {}
        for q, v in zip(train_encoding_std.index.values,
                        train_encoding_std['open_channels'].values):
            if q not in d:
                d[q] = v
        test_values = []
        for q in test[f'quant_{n_bins}'].values:
            test_values.append(d[q])
        test[f'quant_{n_bins}_std'] = test_values
        
        d = {}
        for q, v in zip(train_encoding_var.index.values,
                        train_encoding_var['open_channels'].values):
            if q not in d:
                d[q] = v
        test_values = []
        for q in test[f'quant_{n_bins}'].values:
            test_values.append(d[q])
        test[f'quant_{n_bins}_var'] = test_values

    for n_bins in n_bins_arr:
        train[f'quant_{n_bins}_mean'] = np.zeros(train.shape[0])
        train[f'quant_{n_bins}_std'] = np.zeros(train.shape[0])
        train[f'quant_{n_bins}_var'] = np.zeros(train.shape[0])
    
    # cv loop для train
    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=17)
    for training_index, validation_index in folds.split(train):
        x_train = train.iloc[training_index]
        x_validation = train.iloc[validation_index]
        for n_bins in n_bins_arr:
            column = f'quant_{n_bins}'
            means = x_validation[column].map(x_train.groupby(column).open_channels.mean())
            stds = x_validation[column].map(x_train.groupby(column).open_channels.std())
            vars_ = x_validation[column].map(x_train.groupby(column).open_channels.var())
            
            x_validation[f'quant_{n_bins}_mean'] = means
            x_validation[f'quant_{n_bins}_std'] = stds
            x_validation[f'quant_{n_bins}_var'] = vars_
            
        train.iloc[validation_index] = x_validation