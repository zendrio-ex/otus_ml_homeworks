import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor
from . import get_model
from sklearn.metrics import mean_absolute_error, mean_squared_error


import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.5f}'.format)


def _list_mean(lst: list) -> float:
    '''calculate mean of list elements.'''
    return sum(lst) / len(lst)


def falling_detection(X_train: pd.DataFrame,
                      y_train: pd.Series,
                      X_val: pd.DataFrame,
                      y_val: pd.Series,
                      common_params: dict,
                      param_space: dict,
                      max_evals: int,
                      falling_window_size: int = 2,
                      nec_delta_by_falling: float = 0.0,
                      step=0.1,
                      f_opt=mean_squared_error,
                      min_feats_num: int = 0,
                      importance_treshold: float = 0) -> list:
    """Selector function. By each iters function sear.chs optimal hps and drops features with little importance.
    It stops when find fall by metrics or len(feats) < 1 or iter's number > max_iterations

    Args:
        X_train (pd.DataFrame): dataframe with fitting features.
        y_train (pd.Series): dataframe with fitting targets.
        X_val (pd.DataFrame): dataframe with validating features.
        y_val (pd.Series): dataframe with validating targets.
        common_params (dict): dict with hyperpatams which aren't changed
        param_space (dict): dict with hyperpatams which are changed
        max_evals (int, optional): iteration number for searching hyperparameters.
        falling_window_size (int, optional): window for fall searching. Defaults to 5.
        nec_delta_by_falling (float, optional): delta for fall detection. Defaults to 0.002.
        step (float, optional): number of featurs which will be dropped after iteration. if you put between 0 and 1, get percents. Defaults to 0.1.
        importance_treshold (float, optional): remove feature with thershold < importance_treshold besides step. Defaults to 0.
    Returns:
        list with features
    """
    l_train = []
    l_val = []
    drop_feats = []
    count_feats = []
    l_suspicious_flag = []
    l_detected_flag = []
    l_suspicion_queue = []
    l_max_val_error = []
    l_falling_window_queue = []

    falling_window_queue = []
    suspicion_queue = []
    list_with_imp = []
    
    feats = X_train.columns.to_list()
    
    max_val_error, passed_iters = 0, 0
    suspicion_flag, was_interrupted = False, False
    
    min_feats_num = 0 if min_feats_num is None else min_feats_num
    
    counter = 0
    while len(feats) > min_feats_num:
        counter += 1
        try:
            estimator = get_model.get_model_with_optuna(X_train, y_train, X_val, y_val,
                                                        common_params=common_params,
                                                        param_space=param_space,
                                                        feats=feats,
                                                        max_evals=max_evals,
                                                        f_opt=f_opt)
            imp_mind = pd.DataFrame({'Name': estimator.feature_name_,
                                     'feature_importances': estimator.feature_importances_}).sort_values('feature_importances',
                                                                                                         ascending=False)
        except KeyboardInterrupt:
            print('It was interrupted.')
            was_interrupted = True
            break
        
        error_val = mean_squared_error(y_val, estimator.predict(X_val[feats]))
        l_val.append(error_val)
        l_train.append(mean_squared_error(y_train, estimator.predict(X_train[feats])))
        count_feats.append(len(feats))
        
        # just for analysis
        l_suspicious_flag.append(suspicion_flag)
        if suspicion_flag:
            l_detected_flag.append(_list_mean(suspicion_queue + [error_val]) > _list_mean(falling_window_queue) + nec_delta_by_falling)
        else:
            l_detected_flag.append(False)
        l_suspicion_queue.append(suspicion_queue + [error_val])
        l_falling_window_queue.append(falling_window_queue if suspicion_flag else falling_window_queue + [error_val])
        
        # falling detection
        if len(falling_window_queue) < falling_window_size:
            falling_window_queue.append(error_val)
            l_max_val_error.append(max_val_error)
        elif not suspicion_flag:
            falling_window_queue.append(error_val)
            max_val_error = max(l_val[:-falling_window_size])
            l_max_val_error.append(max_val_error)
            falling_window_queue.pop(0)
            if (2 * falling_window_size < len(l_val)) & (_list_mean(falling_window_queue) < max_val_error):
                print('falling was suspected!')
                suspicion_flag = True
        else:
            suspicion_queue.append(error_val)
            l_max_val_error.append(max_val_error)
            if len(suspicion_queue) == falling_window_size:
                if _list_mean(suspicion_queue) < _list_mean(falling_window_queue) - nec_delta_by_falling:
                    print('falling was detected!')
                    falling_features_num = count_feats[-2 * falling_window_size - 1]
                    drop_feats.append(imp_mind['Name'].to_list())
                    list_with_imp.append(dict(zip(imp_mind['Name'], imp_mind['feature_importances'])))
                    break
                else:
                    print('suspection was failed.')
                    falling_window_queue = suspicion_queue.copy()
                    max_val_error = max(l_val[:-falling_window_size])
                    suspicion_flag = False
                    suspicion_queue = []

        list_with_imp.append(dict(zip(imp_mind['Name'], imp_mind['feature_importances'])))
        cur_drop_feats = imp_mind[imp_mind['feature_importances'] <= importance_treshold]['Name'].to_list()
        imp_mind = imp_mind[imp_mind['feature_importances'] > importance_treshold]

        if step > 0 and step < 1:
            real_step = floor(imp_mind.shape[0] * step) if floor(imp_mind.shape[0] * step) > 0 else 1
        else:
            real_step = step
        print('numbers of removed features: ', real_step)
        cur_drop_feats += imp_mind[-real_step:]['Name'].to_list()
        feats = imp_mind[:-real_step]['Name'].to_list()
        drop_feats.append(cur_drop_feats)
        
        passed_iters += 1
    
    result = {}
    final_df = pd.DataFrame(
        {'n_feats': count_feats,
         'error_train': l_train,
         'error_val': l_val,
         'drop_feats': drop_feats,
         'l_detected_flag': l_detected_flag,
         'l_suspicious_flag': l_suspicious_flag,
         'l_suspicion_queue': l_suspicion_queue,
         'l_max_val_error': l_max_val_error,
         'l_falling_window_queue': l_falling_window_queue,
         }
    )
    
    result['final_df'] = final_df

    if was_interrupted or len(feats) <= min_feats_num:
        falling_features_num = final_df['n_feats'].min()
        result['remained_features_after_iterruption'] = feats
                       
    result['falling_features_num'] = falling_features_num
    return result

    
def get_features_from_drop_feats(df_with_features: pd.DataFrame,
                                 features_num: int,
                                 features_number_column_name: str = 'n_feats',
                                 features_drop_column_name: str = 'drop_feats'
                                 ) -> list:
    """Restores feats which were dropped by iterations in features_number_column_name.

    Args:
        df_with_features (pd.DataFrame): result df from feature selection.
        features_num (int): iterations number which you want to restore in features_drop_column_name
        features_number_column_name (str, optional): column with itaretions Defaults to 'n_feats'.
        features_drop_column_name (str, optional): column with dropped features. Defaults to 'drop'.

    Returns:
        list: with features
    """
    result = []
    
    df_with_features = df_with_features.sort_values(by=features_number_column_name, ascending=True, ignore_index=True)
    for _, row in df_with_features.iterrows():  # Series
        if row[features_number_column_name] > features_num:
            return result
        for feature in row[features_drop_column_name]:
            result.append(feature)
        
    return result


def get_best_features_before_n_feats(df_with_features: pd.DataFrame,
                                     features_num: int,
                                     features_number_column_name: str = 'n_feats',
                                     error_column_name: str = 'error_val',
                                     features_drop_column_name: str = 'drop_feats',
                                     ) -> dict:
    df_with_features = df_with_features[df_with_features[features_number_column_name] <= features_num]
    
    id_by_min = df_with_features[error_column_name].idxmin()
    num_feats_by_min = df_with_features[features_number_column_name][id_by_min]
    return {'num_feats_by_min': num_feats_by_min,
            'features_list': get_features_from_drop_feats(df_with_features,
                                                          features_num=num_feats_by_min,
                                                          features_number_column_name=features_number_column_name,
                                                          features_drop_column_name=features_drop_column_name)}


def get_falling_features_num(df_with_features: pd.DataFrame,
                             falling_window_size: int = 2,
                             nec_delta_by_falling: float = 0,
                             features_number_column_name: str = 'n_feats',
                             error_column_name: str = 'error_val',
                             features_drop_column_name: str = 'drop_feats',
                             ) -> int:
    if df_with_features.shape[1] <= 2 * falling_window_size:
        return df_with_features[features_number_column_name].max()

    feat_nums = sorted(df_with_features[features_number_column_name].to_list())
    idx = 2 * falling_window_size - 1
    
    while idx < len(feat_nums):
        left_mean = df_with_features[(df_with_features[features_number_column_name] <= feat_nums[idx]) &
                                     (df_with_features[features_number_column_name] > feat_nums[idx - falling_window_size])][error_column_name].mean()
        right_mean = df_with_features[(df_with_features[features_number_column_name] <= feat_nums[idx - falling_window_size]) &
                                      (df_with_features[features_number_column_name] > feat_nums[idx - 2 * falling_window_size + 1])][error_column_name].mean()
        if right_mean - left_mean <= nec_delta_by_falling:
            break
        idx += 1
    return feat_nums[idx]


def print_df(data: pd.DataFrame,
             features_num_by_falling: float = None,
             title: str = '',
             x_name: str = 'n_feats',
             xlim: tuple = None,
             ylim: tuple = (0, 100),
             drop_some_first_iterations: int = 0) -> None:
    """Draws graphs with result from spark or pandas functions with searching algorithms.

    Args:
        data (pd.DataFrame): result df
        features_num_by_falling (float, optional): features_num_by_falling from spark or pandas functions with searching algorithms. Defaults to None.
        title (str, optional): Graph's title. Defaults to ''.
        x_name (str, optional): x_name . Defaults to 'n_feats'.
        xlim (tuple, optional): range for x for graph. Defaults to None.
        ylim (tuple, optional): range for y for graph. Defaults to (0.0, 1.0).
        drop_some_first_iterations (int, optional): doesn't screen first iterations. Defaults to 0.
    """
    data = data.tail(data.shape[0] - drop_some_first_iterations)
    data = pd.DataFrame({'n_feats': data['n_feats'].to_list() * 2,
                         'error': data['error_train'].to_list() + data['error_val'].to_list(),
                         'samples': ['train'] * data.shape[0] + ['val'] * data.shape[0]})
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=1.5)
    f, ax = plt.subplots(figsize=(16, 8))
    ax = sns.lineplot(x=x_name, y='error', hue='samples', data=data, linewidth=3.5, marker='o')
    ax.set_title(title)
    
    if xlim is not None:
        plt.xlim(*xlim)
    plt.ylim(*ylim)

    if features_num_by_falling is not None:
        plt.plot([features_num_by_falling, features_num_by_falling], color='r')
    plt.draw()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), fancybox=True, shadow=True, ncol=2)
