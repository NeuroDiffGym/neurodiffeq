import warnings


def warn_deprecate_class(new_class):
    """get a new class name that issues a warning when instantiated
    :param new_class: new class name
    :type new_class: type
    :return: a function that, when called, acts as if it is a class constructor
    :rtype: callable
    """

    def old_class_getter(*args, **kwargs):
        warnings.warn(f"This class name is deprecated, use {new_class} instead")
        return new_class(*args, **kwargs)

    return old_class_getter
