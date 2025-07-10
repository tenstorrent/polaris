#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.utils.common import prod_ints

from typing import Dict, List, Tuple, Self
from functools import lru_cache

def sieve_of_eratosthenes(N: int) -> List[int]:
    """
    find all primes <= N
    """
    sieve = [True] * (N + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, N + 1, i):
                sieve[j] = False
    return [i for i in range(N + 1) if sieve[i]]

def wheel_factorization(n: int) -> Dict[int, int]:
    assert n >= 1, f"ERR wheel_factorization n({n}) < 1"
    if n == 1: return {}

    @lru_cache(maxsize=256)
    def smallest_prime_factor(m: int) -> int:
        """
        wheel factorization: check for by 2, 3, 5 first
        all primes greater than 3 can be expressed as numbers of the form 6k+1, 6k+5
         why? 2 x 3 = 6, numbers co-prime to 6 are of the form 6k+1 and 6k+5,
           because 6k+4, 6k+2 are divisible by 2 and 6k+3 is divisible by 3
           that leaves only 6k+1, 6k+5... for which we check divisibility
        we check for that next upto sqrt(m)
        """
        if m % 2 == 0: return 2
        if m % 3 == 0: return 3
        if m % 5 == 0: return 5

        for i in range(7, int(m ** 0.5) + 1, 6):
            if m % i == 0:
                return i #6k+1
            if m % (i+4) == 0:
                return i+4 #6k+5
        return m

    factors: Dict[int, int] = {}
    while n > 1:
        p = smallest_prime_factor(n)
        if n == p:
            factors[p] = factors.get(p, 0) + 1
            break
        exp = 0
        while n % p == 0:
            exp += 1
            n //= p
        factors[p] = exp
    return factors

class PrimeFactorization:
    def __init__(self, n: int):
        self.num: int = n
        self.factors: Dict[int, int] = wheel_factorization(n)
        assert self.check(), f"Incorrect PrimeFactorization: {self.num} != {self}"

    def number(self) -> int:
        return self.num

    def product(self) -> int:
        return prod_ints(p**e for p,e in self.factors.items())

    def all_divisors(self) -> List[int]:
        """
        returns all the divisors for this number using its prime factorization
        """
        all_prime_factors = [p for p,e in self.items() for i in range(e)]
        divisors = set()
        def backtrack(i, cur):
            if i == len(all_prime_factors):
                if cur:
                    divisors.add(prod_ints(cur))
                return
            backtrack(i+1, cur)
            backtrack(i+1, cur + [all_prime_factors[i]])
        backtrack(0, [])
        return sorted(divisors)

    def check(self) -> bool:
        return self.number() == self.product()

    #useful opertors <=, +, - which simplify usage
    def __le__(self, rhs: Self) -> bool:
        """
        treating PrimeFactorization as a multiset: {p1,p1,...,p1, p2,p2,...p2,...}
        check if rhs is a subset
        """
        assert isinstance(rhs, PrimeFactorization), \
                f"PrimeFactorization rhs-operand= {rhs} type error"
        res = True
        for p,e in rhs.factors.items():
            if p not in self.factors or e > self.factors[p]:
                res = False
                break
        return res

    def __add__(self, rhs: Tuple[int, int]) -> Self:
        assert isinstance(rhs, tuple) and \
                len(rhs) == 2 and \
                all(isinstance(i, int) for i in rhs), \
                f"PrimeFactorization rhs-operand= {rhs} should be of type (int, int)!!"

        p, e            = rhs
        self.factors[p] = self.factors.get(p,0) + e
        self.num        = self.product()

        return self

    def __sub__(self, rhs: Tuple[int, int]) -> Self:
        assert isinstance(rhs, tuple) and \
                len(rhs) == 2 and \
                all(isinstance(i, int) for i in rhs), \
                f"PrimeFactorization rhs-operand= {rhs} should be of type (int, int)!!"

        p, e = rhs
        if p not in self.factors or self.factors[p] < e:
            assert False, "Cannot remove {rhs} from {self}"

        self.factors[p] -= e
        if self.factors[p] == 0: self.factors.pop(p)
        self.num = self.product()

        return self

    #support map interface
    def items       (self     ): return self.factors.items()
    def keys        (self     ): return self.factors.keys()
    def values      (self     ): return self.factors.values()
    def __iter__    (self     ): return iter(self.factors)
    def __getitem__ (self, key): return self.factors[key]
    def __contains__(self, key): return key in self.factors

    def __repr__(self):
        return "PF(" + " * ".join(f"{p}^{e}" for p,e in self.factors.items()) + ")"

    def __str__(self):
        return " * ".join(f"{p}^{e}" for p,e in self.factors.items())

