from typing import TypeVar, Protocol, Any
from typing_extensions import Self

_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)


class SupportsAdd(Protocol[_T_contra, _T_co]):
    def __add__(self, __x: _T_contra) -> _T_co:
        ...


class AddableSelf(Protocol):
    def __add__(self, other: Self) -> Self:
        ...


class SupportsRAdd(Protocol[_T_contra, _T_co]):
    def __radd__(self, __x: _T_contra) -> _T_co:
        ...


class SupportsMul(Protocol[_T_contra, _T_co]):
    def __mul__(self, __x: _T_contra) -> _T_co:
        ...


class SupportsRMul(Protocol[_T_contra, _T_co]):
    def __rmul__(self, __x: _T_contra) -> _T_co:
        ...


class PartialEq(Protocol):
    def __eq__(self, other: object) -> bool:
        ...


from typing import Hashable


class _SupportsSumWithNoDefaultGiven(SupportsAdd[Any, Any], SupportsRAdd[int, Any], Protocol):
    ...


SupportsSumNoDefaultT = TypeVar("SupportsSumNoDefaultT", bound=_SupportsSumWithNoDefaultGiven)
# SumDefaultT = TypeVar("SumDefaultT", bound = SupportsAdd[SupportsSumWithDefaultT,SupportsSumWithDefaultT])
MulT = TypeVar("MulT", bound=SupportsMul)

AddableT1 = TypeVar("AddableT1", bound=SupportsAdd[Any, Any])
AddableT2 = TypeVar("AddableT2", bound=SupportsAdd[Any, Any])

HashableT = TypeVar("HashableT", bound=Hashable)
