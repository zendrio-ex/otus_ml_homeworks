import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _get_model(X_train,
               y_train,
               X_val,
               y_val,
               params):
    
    eval_set = [(X_val, y_val)]
    estimator = lgb.LGBMRegressor(**params)
        
    estimator.fit(X_train, y_train,
                  eval_set=eval_set,
                  early_stopping_rounds=params.get('early_stopping_rounds', None),
                  eval_metric=params.get('eval_metric', 'l2'),
                  sample_weight=params.get('sample_weight', None),
                  verbose=0)
                                
    return estimator


def get_model_with_optuna(X_train,
                          y_train,
                          X_val,
                          y_val,
                          feats,
                          common_params,
                          max_evals,
                          param_space,
                          optuna_algorithm='tpe',
                          f_opt=mean_squared_error,
                          sample_weight=None,
                          **kwargs):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    sampler = {
        'tpe': optuna.samplers.TPESampler,
        'grid': optuna.samplers.GridSampler,
        'random': optuna.samplers.RandomSampler,
    }[optuna_algorithm]
    eval_set = [(X_val[feats], y_val)]
    study = optuna.create_study(direction="maximize", sampler=sampler(),
                                pruner=optuna.pruners.HyperbandPruner() if optuna_algorithm == 'tpe' else optuna.pruners.MedianPruner())
    
    def objective(trial):
        params = {**common_params}
        for suggestion in param_space.keys():
            for hp in param_space[suggestion]:
                params[hp] = getattr(trial, suggestion)(hp, param_space[suggestion][hp][0], param_space[suggestion][hp][1])
        
        model = lgb.LGBMRegressor(**params)
        
        model.fit(X_train[feats], y_train,
                  eval_set=eval_set,
                  early_stopping_rounds=common_params['early_stopping_rounds'] if 'early_stopping_rounds' in common_params else param_space.get('early_stopping_rounds', 10),
                  eval_metric=common_params['eval_metric'] if 'eval_metric' in common_params else param_space.get('eval_metric', 'l2'),
                  verbose=-1)
        
        acc = f_opt(eval_set[0][1], model.predict(eval_set[0][0]))
        return acc
        
    study.optimize(objective, n_trials=max_evals, show_progress_bar=True)
    
    res_tun = {**common_params, **study.best_params}
    estimator = lgb.LGBMRegressor(**res_tun)
    
    estimator.fit(X_train[feats], y_train,
                  eval_set=eval_set,
                  early_stopping_rounds=common_params['early_stopping_rounds'] if 'early_stopping_rounds' in common_params else res_tun.get('early_stopping_rounds', 10),
                  eval_metric=common_params['eval_metric'] if 'eval_metric' in common_params else param_space.get('eval_metric', 'l2'),
                  verbose=-1)
                                
    return estimator
