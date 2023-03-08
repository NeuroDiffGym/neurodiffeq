import dill


def save_globals(path, global_vars):
    assert isinstance(global_vars, dict), "global vars should be a dictionary mapping global names to objects"
    for key in global_vars:
        assert isinstance(key, str) and key.isidentifier(), f"key {key} is not a valid identifier"

    with open(path, 'wb') as f:
        dill.dump(global_vars, f)


def load_globals(path):
    with open(path, 'rb') as f:
        global_vars = dill.load(f)
    assert isinstance(global_vars, dict), "global vars should be a dictionary mapping global names to objects"
    return global_vars
