from __future__ import annotations

import collections
import functools
import itertools
import operator
import random
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
    overload,
)

from typing_extensions import Self

from .helper_types import (
    AddableT1,
    AddableT2,
    HashableT,
    MulT,
    SupportsRichComparison,
    SupportsRichComparisonT,
    SupportsSumNoDefaultT,
)

T = TypeVar("T")
SI_T = TypeVar("SI_T", covariant=True)
V = TypeVar("V")

Predicate: TypeAlias = Callable[[T], bool]

NOFILL = object()


class SI(Generic[SI_T]):
    def __init__(self, it: Iterable[SI_T]) -> None:
        self._it: Iterator[SI_T] = iter(it)

    def __iter__(self) -> Iterator[SI_T]:
        return self._it

    def __next__(self) -> SI_T:
        return next(self._it)

    def map(self: SI[SI_T], fun: Callable[[SI_T], V]) -> SI[V]:
        """
        Applies a callable to each element, yielding the results.
        """
        return type(self)(map(fun, self))

    def count(self) -> int:
        """
        Consumes the iterator, returning the number of elements.
        """
        return sum(1 for _ in self)

    def flat_map(self: SI[SI_T], fun: Callable[[SI_T], Iterable[V]]) -> SI[V]:
        """
        Applies a callable to each element that returns an iterable of values for each element, and yields all the values.
        """
        return type(self)(itertools.chain.from_iterable(map(fun, self)))

    def flatten(self: SI[Iterable[SI_T]]) -> SI[SI_T]:
        return type(self)(itertools.chain.from_iterable(self))  # type:ignore # not sure why it's not typechecking

    def filter(self, pred: Optional[Predicate[SI_T]] = None) -> Self:
        """
        Retains only elements for which pred returns True. If pred isn't provided, filters by truthiness.
        """
        return type(self)(filter(pred, self))

    def reduce(self, fun: Callable[[SI_T, SI_T], SI_T]) -> SI_T:
        """
        Applies a reduction function to the iterator. Raises TypeError if iterator is empty.
        """
        return functools.reduce(fun, self)

    @overload
    def max(self: SI[SupportsRichComparisonT]) -> SupportsRichComparisonT | None:
        ...

    @overload
    def max(self: SI[SI_T], key: Callable[[SI_T], SupportsRichComparison]) -> SI_T | None:
        ...

    def max(self, key=None):  # type:ignore
        if key is None:
            return max(self, default=None)  # type:ignore
        return max(self, key=key, default=None)

    @overload
    def min(self: SI[SupportsRichComparisonT]) -> SupportsRichComparisonT | None:
        ...

    @overload
    def min(self: SI[SI_T], key: Callable[[SI_T], SupportsRichComparison]) -> SI_T | None:
        ...

    def min(self, key=None):  # type:ignore
        if key is None:
            return min(self, default=None)  # type:ignore
        return min(self, key=key, default=None)

    @overload
    def max_(self: SI[SupportsRichComparisonT]) -> SupportsRichComparisonT:
        ...

    @overload
    def max_(self: SI[SI_T], key: Callable[[SI_T], SupportsRichComparison]) -> SI_T:
        ...

    def max_(self: SI, key=None):
        if key is None:
            return max(self)
        return max(self, key=key)

    @overload
    def min_(self: SI[SupportsRichComparisonT]) -> SupportsRichComparisonT:
        ...

    @overload
    def min_(self: SI[SI_T], key: Callable[[SI_T], SupportsRichComparison]) -> SI_T:
        ...

    def min_(self: SI, key=None):
        if key is None:
            return min(self)
        return min(self, key=key)

    # This method is somewhat of an outlier (no variable default??), but that's required for it to be implemented via the builtin sum
    # TODO: modify bounds to detect invalid start for this element type? Might be impossible.
    @overload
    def sum(self: SI[SupportsSumNoDefaultT]) -> Union[SupportsSumNoDefaultT, Literal[0]]:
        ...

    @overload
    def sum(self: SI[AddableT1], start: AddableT2) -> Union[AddableT1, AddableT2]:
        ...

    def sum(self, start=None):
        """
        Returns the sum of a "start" value (default:0) and the elements.
        """
        if start is None:
            return sum(self)  # type:ignore
        return sum(self, start)  # type:ignore

    def nlargest(self: SI[SupportsRichComparisonT], n: int) -> SI[SupportsRichComparisonT]:
        """
        Retains only the n largest elements from self, yields them in ascending order
        """
        lst = sorted(self)
        return type(self)(lst[-n:])

    def nlowest(self: SI[SupportsRichComparisonT], n: int) -> SI[SupportsRichComparisonT]:
        """
        Retains only the n smallest elements from self, yields them in ascending order
        """
        lst = sorted(self)
        return type(self)(lst[:n])

    @overload
    def prod(self: SI[MulT], default: MulT) -> MulT:
        ...

    @overload
    def prod(self: SI[MulT]) -> Union[MulT, Literal[1]]:
        ...

    def prod(self: SI[MulT], default=1):
        """
        Returns the product of elements, or default if there's no elements.
        """
        try:
            return self.reduce(operator.mul)
        except TypeError:
            return default

    def for_each(self, fun: Callable[[SI_T], Any]) -> None:
        """
        Applies a function to each element in the iterator, consuming it.
        """
        for el in self:
            fun(el)

    def enumerate(self: SI[SI_T], start: int = 0) -> SI[Tuple[int, SI_T]]:
        """
        See enumerate from the standard library.
        """
        return type(self)(enumerate(self, start=start))

    def to_list(self) -> list[SI_T]:
        """
        Collects the iterator into a list.
        """
        return list(self)

    def feed(self, fun: Callable[[Self], V]) -> V:
        """
        Returns fun(self).
        """
        return fun(self)

    def sliding_window(self: SI[SI_T], k: int, tooshort_ok: bool = True) -> SI[tuple[SI_T, ...]]:
        """Returns an SI of k-sized sliding windows.
        If there's not enough elements for one window,
        it'll still be yielded if tooshort_ok, otherwise the iterator will be empty.
        """
        return type(self)(self._sliding_window(k=k, tooshort_ok=tooshort_ok))

    def _sliding_window(self, k: int, tooshort_ok: bool = True) -> Iterator[tuple[SI_T, ...]]:
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

    def shifted_pairs(self: SI[SI_T], k: int) -> SI[tuple[SI_T, SI_T]]:
        """
        Returns an iterator of tuples of elements (self[i],self[i+k]).
        There'll only be len(self)-k+1 such tuples.
        """
        lst = self.to_list()
        return type(self)(zip(lst, lst[k:]))

    def groups(self: SI[SI_T], k: int = 2, fillvalue=NOFILL) -> SI[tuple[SI_T, ...]]:
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

    # Here, return type is widened to the default's type.
    @overload
    def find(self: SI[T], pred: Optional[Predicate[T]], default: T) -> T:
        ...

    @overload
    def find(self, pred: Optional[Predicate[SI_T]], default: None) -> Optional[SI_T]:
        ...

    def find(self, pred=None, default=None):
        """
        Returns the first element for which pred returns True, or default if not found.
        If pred isn't provided, filters by truthiness.
        """
        return next(filter(pred, self), default)

    @overload
    def next(self) -> SI_T:
        ...

    @overload
    def next(self, default: V) -> Union[SI_T, V]:
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
    def last(self) -> SI_T:
        ...

    @overload
    def last(self, default: V) -> Union[SI_T, V]:
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

    def any(self, pred: Predicate[SI_T]) -> bool:
        """
        Returns True if there's at least one item for which pred is True.
        Returns False on an empty iterable.
        """
        return any(pred(x) for x in self)

    def all(self, pred: Predicate[SI_T]) -> bool:
        """
        Returns True if there's at no items for which pred is False.
        Returns True on an empty iterable.
        """
        return all(pred(x) for x in self)

    def inspect(self, fun: Callable[[SI_T], Any]) -> Self:
        """
        Applies fun to each element of the iterator as it gets yielded, passing the element on unchanged.
        """
        return self.map(lambda x: (fun(x), x)[1])

    def zip(self: SI[SI_T], other: Iterable[V]) -> SI[Tuple[SI_T, V]]:
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
    def tee(self) -> Tuple[Self, Self]:
        ...

    @overload
    def tee(self, n: Literal[2]) -> Tuple[Self, Self]:
        ...

    @overload
    def tee(self, n: Literal[3]) -> Tuple[Self, Self, Self]:
        ...

    @overload
    def tee(self, n: int = 2) -> Tuple[Self, ...]:
        ...

    def tee(self, n: int = 2) -> Tuple[Self, ...]:
        "Splits the iterator into n independent copies. See itertools.tee."
        return tuple(map(type(self), itertools.tee(self, n)))

    def partition(self, pred: Predicate[SI_T]) -> Tuple[list[SI_T], list[SI_T]]:
        """Consumes the iterator and returns two lists: all elements for which pred is True, and all elements for which pred is False."""
        true = []
        false = []
        for el in self:
            if pred(el):
                true.append(el)
            else:
                false.append(el)
        return true, false

    def group_by(self, key: Callable[[SI_T], V]) -> dict[V, list[SI_T]]:
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

    def unique(self: SI[HashableT]) -> SI[HashableT]:
        """
        Retains only the first appearance of each unique (by equality) element. Requires the elements to be hashable.
        """
        seen = set()

        def allow(el: HashableT) -> bool:
            if el in seen:
                return False
            seen.add(el)
            return True

        return self.filter(allow)

    def value_counts(self: SI[HashableT]) -> collections.Counter[HashableT]:
        """
        Feeds the iterator into collections.Counter, returning a Counter mapping unique elements to their number of occurences.
        Requires the elements to be hashable.
        """
        return self.feed(collections.Counter)

    def cumsums(self: SI[SupportsSumNoDefaultT]) -> SI[SupportsSumNoDefaultT]:
        """
        [1,2,3] -> [1,3,6]
        """
        return type(self)(itertools.accumulate(self))

    @overload
    def sorted(self: SI[SupportsRichComparisonT]) -> SI[SupportsRichComparisonT]:
        ...

    @overload
    def sorted(self: SI[SI_T], key: Callable[[SI_T], SupportsRichComparison]) -> SI[SI_T]:
        ...

    def sorted(self: SI, key=None):
        if key is None:
            return type(self)(sorted(self))
        return type(self)(sorted(self, key=key))
