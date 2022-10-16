from typing import TypeVar, Protocol, Any

_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)


class SupportsAdd(Protocol[_T_contra, _T_co]):
    def __add__(self, __x: _T_contra) -> _T_co:
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


class _SupportsSumWithNoDefaultGiven(SupportsAdd[Any, Any], SupportsRAdd[int, Any], Protocol):
    ...


_SupportsSumNoDefaultT = TypeVar("_SupportsSumNoDefaultT", bound=_SupportsSumWithNoDefaultGiven)
_MulT = TypeVar("_MulT", bound=SupportsMul)

_AddableT1 = TypeVar("_AddableT1", bound=SupportsAdd[Any, Any])
_AddableT2 = TypeVar("_AddableT2", bound=SupportsAdd[Any, Any])
