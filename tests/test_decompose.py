from subline.utils.decompose import sliceable


def test_sliceable():
    from collections import namedtuple

    Foo = sliceable(namedtuple("Foo", "a,b,c,d"))

    foo = Foo(1, 2, 3, 4)
    assert (2,) == foo["b"]
    assert (1, 3) == foo["a c"]
    assert (1, 3) == foo[["a", "c"]]
    assert (1, 3) == foo[[0, 2]]
    assert (2, 3, 4) == foo["b:d"]
    assert foo == (1, 2, 3, 4)
