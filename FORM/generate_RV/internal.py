from functools import partial


def _error_RV(x, rv, mean, std, fixed_params, search_params):
    sp = {s: v for (s, v) in zip(search_params, x)}
    random_var = partial(rv, **fixed_params)(**sp)
    if std is None:
        return [mean - random_var.mean()]
    else:
        return [mean - random_var.mean(), std - random_var.std()]
