import logging

import lightgbm as lgb
import numpy as np
from hyperopt import fmin, hp, atpe, STATUS_OK, STATUS_FAIL, Trials
from sklearn.metrics import f1_score

from default_detection import RANDOM_STATE


def construct_scores(trials):
    """Construct a dictionary from the search result. Has a similar structure as
    the cv_results_ dict from sklearn."""
    scores = {'mean_test_score': [],
              'params': [],
              'auc': [],
              'f1': [],
              'rank_test_score': []}

    for i in range(len(trials.trials)):
        scores['mean_test_score'].append(trials.trials[i]['result']['loss'])
        scores["auc"].append(trials.trials[i]['result']["auc"][-1])
        scores["f1"].append(trials.trials[i]['result']["f1"][-1])
        scores['params'].append(trials.trials[i]['result']['params'])

    order = np.argsort(trials.losses())
    scores['rank_test_score'] = order.argsort() + 1

    return scores


class HyperOptimization(object):
    def __init__(self, lgb_train, features, categorical_features, target):
        self.trials = Trials()
        self.logger = logging.getLogger(__name__)
        self.lgb_train = lgb_train
        self.features = features
        self.target = target
        self.categorical_features = categorical_features

    def process(self, space, trials, algo, max_evals, timeout):
        """Minimize the objective function and return the best parameters."""
        try:
            results = fmin(
                fn=self.objective_function,
                space=space,
                algo=algo,
                max_evals=max_evals,
                timeout=timeout,
                trials=trials
            )
        except Exception as e:
            self.logger.exception("Failure while evaluating minimizing objective")
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return results

    def objective_function(self, params):
        """Objective function to minimize."""

        # Convert parameters to int since hyperopt converts to float values
        params['max_depth'] = int(params['max_depth'])
        params['num_leaves'] = int(params['num_leaves'])

        # Compute cross validation scores with early stopping rounds
        scores = lgb.cv(
            params=params,
            num_boost_round=500,
            train_set=self.lgb_train,
            nfold=3,
            feature_name=self.features,
            categorical_feature=self.categorical_features,
            stratified=True,
            early_stopping_rounds=150,
            feval=self.f1_score_valid,
        )

        # Save the scores
        loss = -1 * scores['auc-mean'][-1]
        variance = scores['auc-stdv'][-1]

        # Return results for this iteration
        results = {
            'loss': loss,
            "f1": scores["f1-mean"],
            "auc": scores["auc-mean"],
            'loss_variance': variance,
            'params': params,
            'status': STATUS_OK
        }

        return results

    def hyper_search(self, max_evals=100, timeout=600, algo=atpe.suggest):
        """Return the best parameters found from hyper search."""
        self.logger.info("Starting hyper-search.")

        # Parameter space for discrete parameters
        integer_params = {
            'max_depth': hp.choice('max_depth', [3, 5, 7, 9, 11, -1]),
            'num_leaves': hp.quniform('num_leaves', 25, 512, 1),
        }

        # Parameter space for continuous parameters
        float_params = {
            'min_split_gain': hp.uniform('min_split_gain', 0.1, 5),
            'subsample': hp.uniform('subsample', 0.5, 0.8),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 0.6),
            'reg_alpha': hp.uniform('reg_alpha', 0, 0.5),
            'reg_lambda': hp.uniform('reg_lambda', 0.5, 10),
            "scale_pos_weight": hp.uniform('scale_pos_weight', 1, 100),
        }

        # Parameters to keep fixed
        fix_params = {
            'learning_rate': 0.01,
            'n_jobs': 8,
            'objective': 'binary',
            'random_state': RANDOM_STATE,
            'verbose': -1,
            "metric": ['auc', 'f1'],
            "first_metric_only": True,
        }

        # Construct the search space and fixed parameters to set on the model
        params = dict()
        params.update(integer_params)
        params.update(float_params)
        params.update(fix_params)

        # Optimize over parameter space
        self.process(space=params, trials=self.trials, algo=algo, max_evals=max_evals, timeout=timeout)
        self.logger.info('Finished hyper-search.')

        # Update with tuned parameter values and increase estimators
        optimized_params = self.trials.best_trial['result']['params']
        optimized_params['n_estimators'] = len(self.trials.best_trial['result']['auc'])

        # Construct sklearn scores in style of cv_results_ from sklearn
        scores = construct_scores(self.trials)

        best_scores = {
            "auc": scores["auc"][np.argmin(scores["rank_test_score"])],
            "f1": scores["f1"][np.argmin(scores["rank_test_score"])]
        }

        return optimized_params, best_scores

    @staticmethod
    def f1_score_valid(y_hat, lgb_data):
        y_true = lgb_data.get_label()
        y_hat = np.where(y_hat < 0.5, 0, 1)
        return 'f1', f1_score(y_true, y_hat), True
