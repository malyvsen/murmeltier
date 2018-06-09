def assert_one(**kwargs):
    arg = None
    for key in kwargs:
        if kwargs[key] is not None:
            if arg is None:
                arg = kwargs[key]
            else:
                raise AssertionError('Provide exactly one of ' + str(set(kwargs.keys())) + ' - provided: ' + str(kwargs))
    return arg


def assert_disjoint(**kwargs):
    union = set()
    total_len = 0
    for arg in kwargs.items():
        if isinstance(arg, dict):
            union |= set(arg.keys())
        else:
            union |= arg
        total_len += len(arg)
    if total_len != len(union):
        raise AssertionError('Should be disjoint: ' + str(kwargs))


def assert_keys(keys, **kwargs):
    for key in kwargs:
        if kwargs[key].keys() is not keys:
            raise AssertionError(key + ' must exactly have keys: ' + str(set(keys)))


def assert_equal_or_none(**kwargs):
    result = None
    for arg in kwargs.items():
        if arg is None:
            continue
        if result is None:
            result = arg
            continue
        if arg is not result:
            raise AssertionError('Shuld be equal or None: ' + kwargs)
    return result
