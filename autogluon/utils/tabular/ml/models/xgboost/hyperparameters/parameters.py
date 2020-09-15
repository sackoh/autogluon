from ....constants import BINARY, MULTICLASS, REGRESSION

DEFAULT_NUM_BOOST_ROUND = 10000


def get_param_baseline(problem_type, num_classes=None):
    if problem_type == BINARY:
        return get_param_binary_baseline()
    elif problem_type == MULTICLASS:
        return get_param_multiclass_baseline(num_classes=num_classes)
    elif problem_type == REGRESSION:
        return get_param_regression_baseline()
    else:
        return get_param_binary_baseline()


def get_param_binary_baseline():
    params = {
        'n_estimators': DEFAULT_NUM_BOOST_ROUND,
        'learning_rate': 0.1,
        'n_jobs': -1,
        'objective': 'binary:logistic'
    }
    return params


def get_param_multiclass_baseline(num_classes):
    params = {
        'n_estimators': DEFAULT_NUM_BOOST_ROUND,
        'learning_rate': 0.1,
        'n_jobs': -1,
        'objective': 'multiclass:softmax',
        'num_class': num_classes,
    }
    return params


def get_param_regression_baseline():
    params = {
        'n_estimators': DEFAULT_NUM_BOOST_ROUND,
        'learning_rate': 0.1,
        'n_jobs': -1,
        'objective': 'reg:squarederror',
    }
    return params
