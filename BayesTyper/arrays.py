"""
besmarts.core.arrays

The core data types that store SMARTS primitives.

NOTE: This module is copied from the Besmarts package, developed by Trevor Gokey, UC Irvine.
https://github.com/trevorgokey/besmarts
"""

from typing import List, Tuple


INF = -1

class bitvec:

    """
    Holds the bit vector of a single primitive as a single integer.
    """

    __slots__ = "v", "maxbits"

    def __init__(self, val=0, maxbits=64):

        self.v: int = val
        self.maxbits: int = maxbits

    @property
    def inv(self):
        return self.v < 0

    @inv.setter
    def inv(self, switch: bool):
        if switch and self.v < 0:
            self.v = ~self.v
        elif (not switch) and self.v >= 0:
            self.v = ~self.v

    def __iter__(self):
        for x in bitvec_on(self):
            b = bitvec(maxbits=self.maxbits)
            b[x] = True
            yield b

    def __getitem__(self, i):

        l = min(len(self), self.maxbits)
        if isinstance(i, slice):

            # [:]
            if i.stop is None and i.start is None and i.step is None:
                return [((self.v >> x) & 1) for x in range(l)]

            start = 0 if i.start is None else i.start
            end = i.stop

            if end is None:
                end = self.maxbits

            if self.maxbits == INF:
                raise IndexError(
                    "Cannot supply an infinite length array "
                    "(input slice had no bounds and maxbits was inf (-1))"
                )

            step = 1 if i.step is None else i.step
            rev = step < 0
            step = abs(step)

            # [i:] and i > maxbits
            if start >= l:
                diff = (end - start + 1) // step
                pad = list([int(self.v < 0)] * diff)
                pad = pad[::-1]
                return pad

            # [i:j:k] and 2**j > v
            if end > l:
                diff = (end - l) // step
                expl = [(self.v >> x) & 1 for x in range(start, l, step)]
                pad = list([int(self.v < 0)] * diff)
                expl += pad
                if rev:
                    expl = expl[::-1]
                return expl

            # [i:j:k] and all within v
            expl = [(self.v >> x) & 1 for x in range(start, end, step)]
            if rev:
                expl = expl[::-1]
            return expl

        # [i]
        elif i >= l:
            return int(self.v < 0)
        else:
            return (self.v >> i) & 1

    def __setitem__(self, i, v: int):

        v: bool = v > 0

        if isinstance(i, slice):

            # [:] = v
            if i.stop is None and i.start is None and i.step is None:
                if v:
                    self.v = -1
                else:
                    self.v = 0
                return

            start = 0 if i.start is None else i.start
            end = int(max(self.maxbits if i.stop is None else i.stop, start))

            if end > self.maxbits:
                end = self.maxbits

            step = 1 if i.step is None else i.step

            # [x:] = v
            if i.stop is None:
                mask = sum(2**i for i in range(start, end, step))
                if v:
                    self.v |= mask
                else:
                    mask = ~mask
                    self.v &= mask

            else:
                # [x:y:z] = v
                mask = sum(2**i for i in range(start, end, step))
                if v:
                    self.v |= mask
                else:
                    mask = ~mask
                    self.v &= mask

        elif isinstance(i, int):
            mask = 2**i
            if v:
                self.v |= mask
            else:
                mask = ~mask
                self.v &= mask

        else:
            raise Exception(
                "Using datatype {} for setitem not supported".format(type(i))
            )

    def __len__(self) -> int:
        if self.v == 0:
            return 0

        l = len(bin(self.v))

        if self.v < 0:
            return l - 3

        return l - 2

    def __repr__(self) -> str:

        v = self.v
        neg = " "

        if v < 0:
            v = -self.v - 1
            neg = "~"

        return neg + "{:>s}".format(bin(v))

    def __hash__(self) -> int:
        return self.v

    def __and__(self, o) -> "bitvec":
        return bitvec_and(self, o)

    def __or__(self, o) -> "bitvec":
        return bitvec_or(self, o)

    def __xor__(self, o) -> "bitvec":
        return bitvec_xor(self, o)

    def __invert__(self) -> "bitvec":
        return bitvec_not(self)

    def __add__(self, o) -> "bitvec":
        return bitvec_or(self, o)

    def __sub__(self, o) -> "bitvec":
        return bitvec_subtract(self, o)

    def __eq__(self, o) -> bool:
        return bitvec_equal(self, o)

    def __ne__(self, o) -> bool:
        return not self == o

    def any(self) -> bool:
        return bitvec_any(self)

    def all(self) -> bool:
        return bitvec_all(self)

    def is_null(self) -> bool:
        return bitvec_is_null(self)

    def on(self) -> List[int]:
        return bitvec_on(self)

    def off(self) -> List[int]:
        return bitvec_off(self)

    def bits(self, maxbits=False) -> int:
        return bitvec_bits(self, maxbits=maxbits)

    def reduce(self) -> int:
        return bitvec_sum(self)

    def clear(self):
        self.v = 0

    def copy(self):
        return bitvec_copy(self)


def bitvec_sum(bv: bitvec) -> int:
    return bv.v


def bitvec_is_inverted(bv: bitvec) -> bool:
    return bv.v < 0


def bitvec_bits(bv: bitvec, maxbits=False) -> int:

    inv = bitvec_is_inverted(bv)
    if inv:
        if maxbits:
            return bv.maxbits
        else:
            return INF

    return len(bitvec_on(bv))


def bitvec_on(bv: bitvec) -> List[int]:
    return [i for i in range(bv.maxbits) if (bv.v >> i) & 1]


def bitvec_off(bv: bitvec) -> List[int]:
    return [i for i in range(bv.maxbits) if not (bv.v >> i) & 1]


def bitvec_explicit_flip(bv: bitvec) -> None:
    bv.v = ~bv.v


def bitvec_is_null(bv: bitvec) -> bool:
    return bv.v == 0


def bitvec_clear(bv: bitvec) -> None:
    bv.v = 0


def bitvec_copy(bv: bitvec) -> bitvec:
    return bitvec(bv.v, bv.maxbits)


def bitvec_all(bv: bitvec) -> bool:
    return bv.v == -1


def bitvec_any(bv: bitvec) -> bool:
    return bv.v != 0


def bitvec_reduce(bv: bitvec) -> int:
    return bv.v


def bitvec_reduce_longest(a: bitvec, b: bitvec) -> Tuple[int, int]:
    return a.v, b.v


def bitvec_not(a: bitvec) -> bitvec:
    return bitvec(~a.v, a.maxbits)


def bitvec_or(a: bitvec, b: bitvec) -> bitvec:
    return bitvec(a.v | b.v, max(a.maxbits, b.maxbits))


def bitvec_and(a: bitvec, b: bitvec) -> bitvec:
    return bitvec(a.v & b.v, min(a.maxbits, b.maxbits))


def bitvec_xor(a: bitvec, b: bitvec) -> bitvec:
    return bitvec(a.v ^ b.v, max(a.maxbits, b.maxbits))


def bitvec_subtract(a: bitvec, b: bitvec) -> bitvec:
    return bitvec(a.v & (a.v ^ b.v), a.maxbits)


def bitvec_equal(a: bitvec, b: bitvec) -> bool:
    return a.v == b.v


def bitvec_subset(a: bitvec, b: bitvec) -> bool:
    return a.v == (a.v & b.v)


def bitvec_superset(a: bitvec, b: bitvec) -> bool:
    return b.v == (a.v & b.v)


array_dtype = type(bitvec)
