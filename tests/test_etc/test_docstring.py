from xtgeo.common.sys import inherit_docstring


class SuperClass:
    def my_function(self):
        """I have a doc string"""


def test_docstring():
    class SubClass(SuperClass):
        @inherit_docstring(inherit_from=SuperClass.my_function)
        def my_function(self):
            pass

    assert SubClass.my_function.__doc__ == "I have a doc string"


def test_new_docstring():
    class SubClass(SuperClass):
        @inherit_docstring(inherit_from=SuperClass.my_function)
        def my_function(self):
            """I have a different doc string"""
            pass

    assert SubClass.my_function.__doc__ == "I have a different doc string"


def test_no_super_docstring():
    class SuperClass:
        def my_function(self):
            pass

    class SubClass(SuperClass):
        @inherit_docstring(inherit_from=SuperClass.my_function)
        def my_function(self):
            """I have a doc string"""
            pass

    assert SubClass.my_function.__doc__ == "I have a doc string"
