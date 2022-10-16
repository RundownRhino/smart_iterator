from __future__ import annotations
import collections
import itertools
import functools
import operator
import random
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Tuple,
    TypeAlias,
    TypeVar,
    Generic,
    Union,
    overload,
)
from typing_extensions import Self

from .helper_types import _SupportsSumNoDefaultT, _MulT, _AddableT1, _AddableT2, _HashableT

T = TypeVar("T")
V = TypeVar("V")

Predicate: TypeAlias = Callable[[T], bool]

NOFILL = object()


class SmartIterator(Generic[T]):
    def __init__(self, it: Iterable[T]):
        self._it = iter(it)

    def __iter__(self) -> Iterator[T]:
        return iter(self._it)

    def __next__(self) -> T:
        return next(self._it)

    def map(self: SI[T], fun: Callable[[T], V]) -> SI[V]:
        """
        Applies a callable to each element, yielding the results.
        """
        return type(self)(map(fun, self))

    def count(self) -> int:
        """
        Consumes the iterator, returning the number of elements.
        """
        return sum(1 for _ in self)

    def flat_map(self: SI[T], fun: Callable[[T], Iterable[V]]) -> SI[V]:
        """
        Applies a callable to each element that returns an iterable of values for each element, and yields all the values.
        """
        return type(self)(itertools.chain.from_iterable(map(fun, self)))

    def filter(self, pred: Optional[Predicate[T]] = None) -> Self:
        """
        Retains only elements for which pred returns True. If pred isn't provided, filters by truthiness.
        """
        return type(self)(filter(pred, self))

    def reduce(self, fun: Callable[[T, T], T]) -> T:
        """
        Applies a reduction function to the iterator. Raises TypeError if iterator is empty.
        """
        return functools.reduce(fun, self)

    # This method is somewhat of an outlier (no variable default??), but that's required for it to be implemented via the builtin sum
    @overload
    def sum(self: SI[_SupportsSumNoDefaultT]) -> Union[_SupportsSumNoDefaultT, Literal[0]]:
        ...

    @overload
    def sum(self: SI[_AddableT1], start: _AddableT2) -> Union[_AddableT1, _AddableT2]:
        ...

    def sum(self, start=None):
        """
        Returns the sum of a "start" value (default:0) and the elements.
        """
        if start is None:
            return sum(self)  # type:ignore
        return sum(self, start)  # type:ignore

    @overload
    def prod(self: SI[_MulT], default: _MulT) -> _MulT:
        ...

    @overload
    def prod(self: SI[_MulT]) -> Union[_MulT, Literal[1]]:
        ...

    def prod(self: SI[_MulT], default=1):
        """
        Returns the product of elements, or default if there's no elements.
        """
        try:
            return self.reduce(operator.mul)
        except TypeError:
            return default

    def for_each(self, fun: Callable[[T], Any]) -> None:
        """
        Applies a function to each element in the iterator, consuming it.
        """
        for el in self:
            fun(el)

    def enumerate(self: SI[T], start: int = 0) -> SI[Tuple[int, T]]:
        """
        See enumerate from the standard library.
        """
        return type(self)(enumerate(self, start=start))

    def to_list(self) -> list[T]:
        """
        Collects the iterator into a list.
        """
        return list(self)

    def feed(self, fun: Callable[[Self], V]) -> V:
        """
        Returns fun(self).
        """
        return fun(self)

    def sliding_window(self: SI[T], k: int, tooshort_ok: bool = True) -> SI[tuple[T, ...]]:
        """Returns an SI of k-sized sliding windows.
        If there's not enough elements for one window,
        it'll still be yielded if tooshort_ok, otherwise the iterator will be empty.
        """
        return type(self)(self._sliding_window(k=k, tooshort_ok=tooshort_ok))

    def _sliding_window(self, k: int, tooshort_ok: bool = True) -> Iterator[tuple[T, ...]]:
        if k < 1:
            raise ValueError(f"k must be >=1, was {k}")
        queue = collections.deque(maxlen=k)
        it = self._it
        for el in itertools.islice(it, 0, k):
            queue.append(el)
        if len(queue) < k:
            if tooshort_ok:
                yield tuple(queue)
            return
        yield tuple(queue)
        for el in it:
            queue.append(el)  # which pops the oldest one
            yield tuple(queue)

    def shifted_pairs(self: SI[T], k: int) -> SI[tuple[T, T]]:
        """
        Returns an iterator of tuples of elements (self[i],self[i+k]).
        There'll only be len(self)-k+1 such tuples.
        """
        lst = self.to_list()
        return type(self)(zip(lst, lst[k:]))

    def groups(self: SI[T], k: int = 2, fillvalue=NOFILL) -> SI[tuple[T, ...]]:
        """
        ABCD -> AB, CD (for k=2)
        Returns an SI of tuples.
        If fillvalue is specified, missing elements will be filled with it; otherwise
        a possible unfull tuple will not be yielded.
        """
        it = self._it
        if fillvalue is NOFILL:
            return type(self)(zip(*([it] * k)))
        else:
            return type(self)(itertools.zip_longest(*([it] * k), fillvalue=fillvalue))

    @overload
    def find(self, pred: Optional[Predicate[T]], default: T) -> T:
        ...

    @overload
    def find(self, pred: Optional[Predicate[T]], default: None) -> Optional[T]:
        ...

    def find(self, pred=None, default=None):
        """
        Returns the first element for which pred returns True, or default if not found.
        If pred isn't provided, filters by truthiness.
        """
        return next(filter(pred, self), default)

    @overload
    def next(self) -> T:
        ...

    @overload
    def next(self, default: V) -> Union[T, V]:
        ...

    def next(self, default=NOFILL):
        """
        Returns the first element of the iterator.
        If iterator is empty, returns default if provided, otherwise raises StopIteration.
        """
        if default is NOFILL:
            return next(self._it)
        return next(self._it, default)

    @overload
    def last(self) -> T:
        ...

    @overload
    def last(self, default: V) -> Union[T, V]:
        ...

    def last(self, default=NOFILL):
        """
        Consumes the iterator and returns its last element.
        If iterator is empty, returns default if provided, otherwise raises StopIteration.
        """
        d = collections.deque(maxlen=1)
        d.extend(self)
        if not d:
            if default is NOFILL:
                raise StopIteration
            return default
        return d.pop()

    def reversed(self) -> Self:
        """
        Reverses the iterator's order. This is done by first collecting it into a list.
        """
        lst = self.to_list()
        lst.reverse()
        return type(self)(lst)

    def shuffled(self, random_fun: Optional[Callable[[], float]] = None) -> Self:
        """
        Yields the elements in random order. This is done by first collecting the iterator into a list.
        If provided, random_fun is used as the randomness source for random.shuffle.
        """
        lst = self.to_list()
        random.shuffle(lst, random=random_fun)
        return type(self)(lst)

    def any(self, pred: Predicate[T]) -> bool:
        """
        Returns True if there's at least one item for which pred is True.
        Returns False on an empty iterable.
        """
        return any(pred(x) for x in self)

    def all(self, pred: Predicate[T]) -> bool:
        """
        Returns True if there's at no items for which pred is False.
        Returns True on an empty iterable.
        """
        return all(pred(x) for x in self)

    def inspect(self, fun: Callable[[T], Any]) -> Self:
        """
        Applies fun to each element of the iterator as it gets yielded, passing the element on unchanged.
        """
        return self.map(lambda x: (fun(x), x)[1])

    def zip(self: SI[T], other: Iterable[V]) -> SI[Tuple[T, V]]:
        return type(self)(zip(self, other))

    def unzip(self: SI[Tuple[T, V]]) -> Tuple[SI[T], SI[V]]:
        """
        "Unzips" an iterator of tuples into a tuple of iterators.
        """
        lefts, rights = self._unzip()  # type:ignore
        return type(self)(lefts), type(self)(rights)  # type:ignore

    def unzip_lists(self: SI[Tuple[T, V]]) -> Tuple[list[T], list[V]]:
        """
        Like unzip, but collects the returned iterators to lists.
        """
        ls = []
        rs = []
        for l, r in self:
            ls.append(l)
            rs.append(r)
        return ls, rs

    def _unzip(self: SI[Tuple[T, V]]) -> Tuple[Iterator[T], Iterator[V]]:
        lefts_queue: collections.deque[T] = collections.deque()
        rights_queue: collections.deque[V] = collections.deque()
        it = self._it

        def eat():
            l, r = next(it)
            lefts_queue.append(l)
            rights_queue.append(r)

        def lefts() -> Iterator[T]:
            while True:
                if not lefts_queue:
                    eat()
                yield lefts_queue.popleft()

        def rights():
            while True:
                if not rights_queue:
                    eat()
                yield rights_queue.popleft()

        return lefts(), rights()

    @overload
    def tee(
        self, n: Literal[1]
    ) -> Tuple[Self,]:
        ...

    @overload
    def tee(self, n: Literal[2]) -> Tuple[Self, Self]:
        ...

    @overload
    def tee(self, n: Literal[3]) -> Tuple[Self, Self, Self]:
        ...

    def tee(self, n: int = 2) -> Tuple[Self, ...]:
        "Splits the iterator into n independent copies. See itertools.tee."
        return tuple(map(type(self), itertools.tee(self, n)))

    def partition(self, pred: Predicate[T]) -> Tuple[list[T], list[T]]:
        """Consumes the iterator and returns two lists: all elements for which pred is True, and all elements for which pred is False."""
        true = []
        false = []
        for el in self:
            if pred(el):
                true.append(el)
            else:
                false.append(el)
        return true, false

    def group_by(self, key: Callable[[T], V]) -> dict[V, list[T]]:
        """Consumes the iterator and returns a dict mapping a value to all elements such that key(element) == value."""
        groups = {}
        for el in self:
            val = key(el)
            groups.setdefault(val, []).append(el)
        return groups

    def cycle(self) -> Self:
        """
        Repeats an iterator endlessly. See itertools.cycle.
        """
        return type(self)(itertools.cycle(self))

    def unique(self: SI[_HashableT]) -> SI[_HashableT]:
        """
        Retains only the first appearance of each unique (by equality) element. Requires the elements to be hashable.
        """
        seen = set()

        def allow(el: _HashableT) -> bool:
            if el in seen:
                return False
            seen.add(el)
            return True

        return self.filter(allow)

    def value_counts(self: SI[_HashableT]) -> collections.Counter[_HashableT]:
        """
        Feeds the iterator into collections.Counter, returning a Counter mapping unique elements to their number of occurences.
        Requires the elements to be hashable.
        """
        return self.feed(collections.Counter)


SI: TypeAlias = "SmartIterator[T]"
