def trimmed_dict(dict, keys):
    return {key: dict[key] for key in dict if key in keys}
