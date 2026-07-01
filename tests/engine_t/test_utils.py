"""Tests for utility functions in src.utils.utils."""

import pytest

from src.engine.utils.decorators import track_provided_args


class TestTrackProvidedArgs:
    """Test suite for the @track_provided_args decorator."""

    def test_tracks_only_positional_args(self):
        """Ensures only provided positional arguments are tracked."""

        # Arrange
        class MyClass:
            @track_provided_args
            def __init__(self, a, b, c=3):
                pass

        # Act
        instance = MyClass(1, 2)

        # Assert
        assert hasattr(instance, "_provided_args")
        assert instance._provided_args == {"a", "b"}

    def test_tracks_only_keyword_args(self):
        """Ensures only provided keyword arguments are tracked."""

        # Arrange
        class MyClass:
            @track_provided_args
            def __init__(self, a=1, b=2, c=3):
                pass

        # Act
        instance = MyClass(b=20, c=30)

        # Assert
        assert hasattr(instance, "_provided_args")
        assert instance._provided_args == {"b", "c"}

    def test_tracks_mixed_positional_and_keyword_args(self):
        """Ensures a mix of positional and keyword arguments are tracked."""

        # Arrange
        class MyClass:
            @track_provided_args
            def __init__(self, a, b, c=3, d=4):
                pass

        # Act
        instance = MyClass(1, b=20, d=40)

        # Assert
        assert hasattr(instance, "_provided_args")
        assert instance._provided_args == {"a", "b", "d"}

    def test_tracks_no_args_when_all_are_defaults(self):
        """Ensures no arguments are tracked when all defaults are used."""

        # Arrange
        class MyClass:
            @track_provided_args
            def __init__(self, a=1, b=2):
                pass

        # Act
        instance = MyClass()

        # Assert
        assert hasattr(instance, "_provided_args")
        assert instance._provided_args == set()

    def test_tracks_self_when_passed_positionally(self):
        """Correctly tracks 'self' when the first positional arg is passed."""

        # Arrange
        class MyClass:
            @track_provided_args
            def __init__(self, a, b=2):
                pass

        # Act
        instance = MyClass(10)

        # Assert
        assert instance._provided_args == {"a"}

    def test_handles_self_passed_as_keyword_correctly(self):
        """Correctly tracks 'self' even if passed as a keyword (unusual but
        possible)."""

        # Arrange
        class MyClass:
            @track_provided_args
            def __init__(self, a=1, b=2):
                pass

        # Act
        instance = MyClass(a=10)
        # This is not standard Python, but we test the decorator's logic
        MyClass.__init__(self=instance, b=20)

        # Assert
        assert instance._provided_args == {"b"}

    def test_does_not_fail_on_standalone_function_without_self(self):
        """Ensures the decorator does not raise an error on a function with no
        'self'."""

        # Arrange
        @track_provided_args
        def my_function(a, b=2):
            # This function is decorated, but since it's not a method,
            # the decorator won't be able to set _provided_args on an instance.
            # The test is to ensure it doesn't crash.
            _ = a, b

        # Act & Assert
        try:
            my_function(1)
        except Exception as e:
            pytest.fail(f"Decorator failed on a standalone function: {e}")

    def test_overwriting_default_with_same_value_is_tracked(self):
        """Ensures an argument is tracked if explicitly passed, even if it's the
        default value."""

        # Arrange
        class MyClass:
            @track_provided_args
            def __init__(self, a=1, b=2):
                pass

        # Act
        instance = MyClass(a=1)

        # Assert
        assert instance._provided_args == {"a"}

    def test_handles_args_and_kwargs_in_signature(self):
        """Ensures decorator works with methods that accept *args and **kwargs."""

        # Arrange
        class MyClass:
            @track_provided_args
            def __init__(self, a, *args, b=2, **kwargs):
                self.a = a
                self.args = args
                self.b = b
                self.kwargs = kwargs

        # Act
        instance = MyClass(1, 100, 200, b=20, c=30, d=40)

        # Assert
        # The decorator correctly identifies 'a' and 'b' as provided.
        # It also tracks 'c' and 'd' from **kwargs.
        # The positional arguments that fall into *args are not named, so they are not
        # tracked.
        assert instance._provided_args == {"a", "b", "c", "d"}
        assert instance.a == 1
        assert instance.args == (100, 200)
        assert instance.b == 20
        assert instance.kwargs == {"c": 30, "d": 40}

    def test_tracks_args_across_inheritance_chain(self):
        """Ensures arguments from both base and subclass constructors are tracked."""

        # Arrange
        class BaseClass:
            @track_provided_args
            def __init__(self, base_arg, base_kwarg="default"):
                self.base_arg = base_arg
                self.base_kwarg = base_kwarg

        class SubClass(BaseClass):
            @track_provided_args
            def __init__(self, sub_arg, sub_kwarg="default", **kwargs):
                super().__init__(**kwargs)
                self.sub_arg = sub_arg
                self.sub_kwarg = sub_kwarg

        # Act
        # Instantiate the subclass, passing arguments for both base and sub constructors
        instance = SubClass(sub_arg=1, base_arg=100, base_kwarg="provided")

        # Assert
        # The decorator should aggregate arguments from both __init__ calls.
        # The order of __init__ calls matters. Here, super().__init__ is called first.
        assert hasattr(instance, "_provided_args")
        assert instance._provided_args == {"sub_arg", "base_arg", "base_kwarg"}
