import pytest
import numpy as np
import pandas as pd
import sys

import os 
dir_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
sys.path.append(f'{dir_path}/..')
sys.path.append(f'{dir_path}/backend')
sys.path.append(f'{dir_path}/../backend')

import sglm
import sglm_ez
import sglm_cv

from sklearn.linear_model import ElasticNet


def test_integration():
    epsilon = 0.01

    X_tmp = pd.DataFrame(np.arange(200).reshape((100, 2)), columns=['A', 'B'])
    X_tmp['B'] = (X_tmp['B']-1) * 2 + 1
    # X_tmp['B'] = np.random.choice(20, size=10)
    X_tmp = sglm_ez.timeshift_cols(X_tmp, ['A'], pos_order=2)
    X_tmp = sglm_ez.diff_cols(X_tmp, ['A', 'B'])
    # X_tmp = sglm_ez.setup_autoregression(X_tmp, ['B'], 4)
    X_tmp = X_tmp.dropna()
    print(X_tmp)
    
    prediction_cols = ['A']
    response_col = 'B'

    glm = sglm_ez.fit_GLM(X_tmp[prediction_cols], X_tmp[response_col], alpha=0.1)
    pred = glm.predict(X_tmp[prediction_cols])
    mse = np.mean((X_tmp[response_col] - pred)**2)

    print(mse)

    cv_idx = sglm_ez.cv_idx_by_timeframe(X_tmp, y=None, num_folds=None, timesteps_per_bucket=20)    
    # print(cv_idx)

    # Step 1: Create a dictionary of lists for these relevant keywords...
    kwargs_iterations = {
        # # 'reg_lambda': [0.0001],
        # 'reg_lambda': [0, 0.01, 0.1, 1.0],
        # 'alpha': [0, 0.01, 0.1, 1.0],
        'alpha': reversed([0.1, 1.0, 10.0]),
        'l1_ratio': [0.1, 0.5, 0.9],
        'fit_intercept': [True, False]
    }

    # Step 2: Create a dictionary for the fixed keyword arguments that do not require iteration...
    kwargs_fixed = {
        'max_iter': 10000,
        # 'score_metric': 'deviance'
    }

    # Step 3: Generate iterable list of keyword sets for possible combinations
    glm_kwarg_lst = sglm_cv.generate_mult_params(kwargs_iterations, kwargs_fixed)
    best_score, best_params, best_model = sglm_ez.simple_cv_fit(X_tmp[prediction_cols], X_tmp[response_col], cv_idx, glm_kwarg_lst, model_type='Normal')

    print(best_score, best_params)
    print(best_model.model)
    print(best_model.coef_, best_model.intercept_)

    sklr = ElasticNet(alpha=0, l1_ratio=0, fit_intercept=True)
    sklr.fit(X_tmp[prediction_cols], X_tmp[response_col])
    
    print(sklr.coef_, sklr.intercept_)


    assert(np.abs(sklr.intercept_ - best_model.intercept_) < epsilon)
    assert(np.all(np.abs(sklr.coef_ - best_model.coef_) < epsilon))


if __name__ == '__main__':
    test_integration()
