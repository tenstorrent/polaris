#!/usr/bin/env python
#SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Iterator, Optional, Iterable
from dataclasses import dataclass
from enum import Enum, auto


#part of memory allocation system, linked to soc_descriptor YAML
#used by "program" module for kernel placement validation
class AllocCoreType(Enum):
    DISPATCH           = auto()
    STORAGE_ONLY       = auto() #Allocates multiple banks per storage cores
    COMPUTE_ONLY       = auto() #runs dispatch kernels
    COMPUTE_AND_STORE  = auto() #Regular Tensix Cores
    INVALID            = auto()

#from umd/device/tt_core_coordinates
class CoreType(Enum):
    ARC         = auto()
    DRAM        = auto()
    ACTIVE_ETH  = auto()
    IDLE_ETH    = auto()
    PCIE        = auto()
    ROUTER_ONLY = auto()
    SECURITY    = auto()

    #part of coordinate system framework used to manage PEs on BH devices
    # specifically at following physical coordinates: (8-10, 8-4, 8-8, 8-6)
    # takes care of:
    #    physical mapping, virtual coordinates, core-management, NOC1 mapping
    # not there for WH devices
    L2CPU       = auto()

    #TENSIX      = auto() -- incompatibility btw ttnn.CoreType & UMD??
    #UNKNOWN     = auto() -- incompatibility btw ttnn.CoreType & UMD??

    #kept for compatibility with soc_descriptor but not used later on
    HARVESTED   = auto()
    ETH         = auto()
    WORKER      = auto()


    @classmethod
    def enumvalue(cls, s:str):
        return CoreType[s.upper()]

    @property
    def cname(self)->str:
        return self.name.lower()

class CoreCoord:
    def __init__(self, *args):
        if isinstance(args[0], (tuple, list)):
            args = args[0] #type: ignore

        assert isinstance(args, (list, tuple)) and len(args) == 2 and \
                all([isinstance(x, int) and x >= 0 for x in args]), \
                f"CoreCoord should a non-negative xy-pair; Illegal inputs ={args} to constructor"

        self.x: int = args[0]
        self.y: int = args[1]

        return

    def __eq__(self, other): return self.x == other.x and self.y == other.y
    def __lt__(self, other): return self.x < other.x or (self.x == other.x and self.y < other.y)
    def __le__(self, other): return self < other or self == other
    def __str__(self)      : return f"({self.x:2d}, {self.y:2d})"

    def __getitem__(self, index):
        assert (0 <= index <= 1), f"Illegal index= {index}"
        val = self.x if index == 0 else self.y
        return val

    def __setitem__(self, index): raise AssertionError("CoreCoord is immutable!!")
    def __delitem__(self, index): raise AssertionError("CoreCoord is immutable!!")

class CoreRange:
    def __init__(self, start: CoreCoord, end: CoreCoord):
        self.start_coord: CoreCoord = start
        self.end_coord  : CoreCoord = end
        assert self.num_cores() > 0, f"CoreRange: {self} not valid!! num_cores={self.num_cores()}"
        return

    def contains(self, coord: CoreCoord) -> bool:
        return (self.start_coord.x <= coord.x <= self.end_coord.x and
                self.start_coord.y <= coord.y <= self.end_coord.y)

    def num_cores(self) -> int:
        return ((self.end_coord.x - self.start_coord.x + 1) *
                (self.end_coord.y - self.start_coord.y + 1))

    def check_overlap(self, other) -> bool:
        is_R = self.start_coord.x > other.end_coord.x #self completely on right
        is_L = self.end_coord.x < other.start_coord.x #self completely on left
        is_B = self.start_coord.y < other.end_coord.y #self completely below
        is_A = self.end_coord.y > other.start_coord.y #self completely above

        if any([is_R, is_L, is_B, is_A]):
               res = False
        else:
            res = True

        return res

    def __iter__(self) -> Iterator[CoreCoord]:
        for yy in range(self.start_coord.y, self.end_coord.y + 1):
            for xx in range(self.start_coord.x, self.end_coord.x + 1):
                yield (CoreCoord(xx, yy))

    def __str__(self):
        return f"CoreRange({self.start_coord} -> {self.end_coord})"

class CoreRangeSet:
    """A collection of non-overlapping core ranges."""
    def __init__(self, ranges: Iterable[CoreRange]):
        self.ranges = list(ranges)
        #check for overlap between all upper triangular pair matrix
        #because overlap(x,y) = overlap(y,x)
        for i, range1 in enumerate(self.ranges):
            for j, range2 in enumerate(self.ranges[i+1:], i+1):
                if range1.check_overlap(range2):
                    raise ValueError(f"Overlapping ranges: {range1} and {range2}")
        return

    def contains(self, coord: CoreCoord) -> bool:
        return any(r.contains(coord) for r in self.ranges)

    def num_cores(self) -> int:
        return sum(r.num_cores() for r in self.ranges)

    def bounding_box(self) -> Optional[CoreRange]:
        """Return smallest rectangle containing all ranges"""
        if not self.ranges:
            return None

        min_x = min(r.start_coord.x  for r in self.ranges)
        min_y = min(r.start_coord.y  for r in self.ranges)
        max_x = max(r.end_coord.x    for r in self.ranges)
        max_y = max(r.end_coord.y    for r in self.ranges)

        return CoreRange(CoreCoord(min_x, min_y), CoreCoord(max_x, max_y))

    def __iter__(self) -> Iterator[CoreCoord]:
        for range_obj in self.ranges:
            yield from range_obj

CoreGrid = CoreRangeSet

if __name__ == '__main__':
    testcases = [
        ( (0, 0), (9,  9), ( 5,  5), (14, 14), True ),
        ( (0, 10), (10,  0), (15,  5), (25, 15), False),
        ( (0, 10), (10,  0), ( 0, 20), (10, 15), False),
        ( (0, 10), (10,  0), (10, 10), (20,  0), False),
        ( (0, 10), (10,  0), ( 0,  0), (10, 10), False),
        ( (2,  8), ( 8,  2), ( 3,  7), ( 7,  3), True ),
        ( (0, 10), (10,  0), ( 0, 10), (10,  0), True ),
        ( (0, 10), (10,  0), (10,  0), (20, 10), False),
        ( (5, 10), ( 5, 15), ( 4,  5), ( 6,  5), False),
    ]

    for test_no,(x0,y0,x1,y1,expected) in enumerate(testcases):
        r1 = CoreRange(CoreCoord(x0), CoreCoord(y0))
        r2 = CoreRange(CoreCoord(x1), CoreCoord(y1))
        print(r1, r2, r1.num_cores(), r2.num_cores())
        exit(0)
        #overlap = r1.check_overlap(r2)
        #print(test_no, r1, r1.num_cores(), r2, r2.num_cores(), overlap, expected)
        #assert overlap == expected
