import inspect
from functools import wraps


def track_provided_args(func):
    """A decorator to track which arguments were explicitly passed to a method.

    This decorator is intended for use on class `__init__` methods. It inspects
    the arguments passed during instantiation and attaches a `set` named
    `_provided_args` to the instance (`self`). This set contains the names of
    all arguments that were explicitly provided by the caller, even if the
    explicitly passed value is the same as the default.

    This is useful for implementing complex initialization logic where the
    behavior depends on whether a parameter was set by the user or fell back
    to a default. For example, determining precedence between two conflicting
    parameters like `gain_db` and `amplitude`.

    Attributes:
        _provided_args (set[str]): A set of names of the arguments that were
            explicitly passed to the decorated method. This is attached to the
            first argument of the decorated method, which is assumed to be the
            class instance (`self`).

    Example:
        >>> class Oscillator:
        ...     @track_provided_args
        ...     def __init__(self, frequency=440, gain_db=-12):
        ...         pass
        ...
        >>> # Instantiate with only one argument
        >>> osc = Oscillator(frequency=880)
        >>>
        >>> # The decorator attaches the `_provided_args` set to the instance
        >>> assert osc._provided_args == {'frequency'}
        >>>
        >>> # Instantiate with both arguments
        >>> osc2 = Oscillator(frequency=440, gain_db=-6)
        >>> assert osc2._provided_args == {'frequency', 'gain_db'}

    Note:
        This decorator assumes the first argument of the decorated function is the
        class instance (`self`) to which `_provided_args` will be attached. It
        will not raise an error on standalone functions but will have no effect,
        as there is no instance to modify.
    """
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Bind the provided arguments to the function signature.
        bound_args = sig.bind_partial(*args, **kwargs)

        # Start with the names of arguments that were explicitly passed.
        provided_args = set(bound_args.arguments.keys())

        # If there's a **kwargs parameter, its name will be in the set.
        # We need to remove it and add the actual keys from the kwargs dict.
        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                if param.name in provided_args:
                    provided_args.remove(param.name)
                    provided_args.update(kwargs.keys())
            # If there's a *args parameter, its name will be in the set.
            # We should remove it as it doesn't represent a single named argument.
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                if param.name in provided_args:
                    provided_args.remove(param.name)

        # Exclude 'self' if it was captured.
        if 'self' in provided_args:
            provided_args.remove('self')

        # Attach the set to the instance.
        instance = None
        if args and hasattr(args[0], '__dict__'):
            instance = args[0]
        elif 'self' in kwargs:
            instance = kwargs['self']

        if instance:
            # Always overwrite the set with the arguments from the current call.
            # This ensures that if __init__ is called multiple times, we only
            # track the arguments from the most recent call.
            instance._provided_args = provided_args

        # Call the original function.
        return func(*args, **kwargs)

    return wrapper
