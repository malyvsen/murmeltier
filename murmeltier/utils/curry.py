from copy import deepcopy


def curry(function, **kwargs):
    def result(**final_kwargs):
        kwargs_to_pass = deepcopy(kwargs)
        kwargs_to_pass.update(final_kwargs)
        return function(**kwargs_to_pass)
    return result
