import warnings


def warn_deprecate_class(new_class_name):
    """get a new class name that issues a warning when instantiated
    :param new_class_name: new class name
    :type new_class_name: type
    :return: a function that, when called, acts as if it is a class constructor
    :rtype: callable
    """

    def old_class_name(*args, **kwargs):
        warnings.warn(f"This class is deprecated, use {new_class_name.__class__.__name__} instead")
        return new_class_name(*args, **kwargs)

    return old_class_name
