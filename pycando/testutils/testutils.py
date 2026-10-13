#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import time


# %%
def parse_with(parser, params, *flags):
    """"""
    res = []
    for k, v in params.items():
        res.append(f'--{k}')
        res.append(f'{str(v)}')
    for k in flags:
        res.append(f'--{k}')
    return parser.parse_args(res)


# %%
class Times():
    """
    registering time on init;
    three different 'types' of time (as defined in `time` module)
    """
    def __init__(self):
        self.process_time = time.process_time()
        self.process_time_ns = time.process_time_ns()
        self.perf_counter = time.perf_counter()

    def __sub__(self, other):
        dpt = self.process_time - other.process_time
        dptns = self.process_time_ns - other.process_time_ns
        dpc = self.perf_counter - other.perf_counter
        return DTimes(dpt, dptns, dpc)

    def __str__(self):
        ss = f" process time:        {self.process_time}\n" + \
             f" process time (ns):   {self.process_time_ns}\n" + \
             f" performance counter: {self.perf_counter}\n"
        return ss

    def __repr__(self):
        return self.__str__()


class DTimes():
    """
    time delta for Times
    """
    def __init__(self, dpt, dptns, dpc):
        self.process_time = dpt
        self.process_time_ns = dptns
        self.perf_counter = dpc

    def __str__(self):
        ss = "elapsed\n" + \
            f" process time:        {self.process_time}\n" + \
            f" process time (ns):   {self.process_time_ns}\n" + \
            f" performance counter: {self.perf_counter}\n"
        return ss

    def __add__(self, other):
        pt = self.process_time + other.process_time
        ptns = self.process_time_ns + other.process_time_ns
        pc = self.perf_counter + other.perf_counter
        return DTimes(pt, ptns, pc)

    def __sub__(self, other):
        dpt = self.process_time - other.process_time
        dptns = self.process_time_ns - other.process_time_ns
        dpc = self.perf_counter - other.perf_counter
        return DTimes(dpt, dptns, dpc)

    def __repr__(self):
        return self.__str__()


class Timer():
    """
    registers time on init:
    > timer = Timer()
    registers time on stop:
    > timer.stop()
    and calculates time delta:
    > timer.diff     # displays time delta for all three 'time types'
    > timer.elapsed  # alias for  timer.diff
    > timer          # displays timer.start, timer.stop, timer.diff
    consecutive stops possible
    -- time.start does not change thus time.diff always wrt to time.start at init.
    """
    def __init__(self):
        self.start = Times()

    def stop(self):
        self.stop = Times()
        self.diff = self.stop - self.start
        self.elapsed = self.diff  # alias

    def __str__(self):
        ss = "start\n" + self.start.__str__() + \
             "stop\n" + self.stop.__str__() + \
             self.diff.__str__()
        return ss

    def __repr__(self):
        return self.__str__()


# %%
