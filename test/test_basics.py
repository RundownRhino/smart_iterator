from typing_extensions import Self
from dataclasses import dataclass
import pytest
from smart_iterator import SI


def test_basics():
    assert list(map(list, SI([1, 2, 3]).tee(4))) == [[1, 2, 3]] * 4
    assert SI([1, 2, 3]).last() == 3


@dataclass
class MyElem:
    val: int

    def __add__(self, other: Self) -> Self:
        return MyElem(self.val + other.val)


@dataclass
class MyStart:
    pow: int

    def __add__(self, other: MyElem) -> MyElem:
        return MyElem(self.pow**other.val)


def test_sum_ok():
    x = SI([2, 3, 4]).map(MyElem).sum(start=MyStart(5))
    assert x == MyElem(32)  # typechecks, as it should


def test_sum_fails():
    with pytest.raises(TypeError):
        a = (
            SI([1, 2, 3]).map(MyElem).sum(start="5")
        )  # typechecks, even though it really shouldn't - str.__add__ exists, but not for the right type.
