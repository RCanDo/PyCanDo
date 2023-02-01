#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import time


# %%
class Times():
    """
    registering time on init;
    three different 'types' of time (as defined in `time` module)

    https://docs.python.org/3/library/time.html
    https://docs.python.org/3/library/time.html#time.localtime
    https://docs.python.org/3/library/time.html#time.process_time
    https://docs.python.org/3/library/time.html#time.perf_counter

    """
    def __init__(self):
        self.time = time.time()
        self.process_time = time.process_time()
        self.perf_counter = time.perf_counter()

    def __sub__(self, other):
        lt = self.time - other.time
        pt = self.process_time - other.process_time
        pc = self.perf_counter - other.perf_counter
        return DTimes(lt, pt, pc)

    def __str__(self):
        isotime = time.strftime("%Y-%m-%d %H:%M:%S.", time.gmtime(self.time))
        ss = f" local time:          {isotime}\n" + \
             f" process time:        {self.process_time}\n" + \
             f" performance counter: {self.perf_counter}\n"
        return ss

    def __repr__(self):
        return self.__str__()


class DTimes():
    """
    time delta for Times
    """
    def __init__(self, dlt, dpt, dpc):
        self.time = dlt
        self.process_time = dpt
        self.perf_counter = dpc

    def __str__(self):
        isotime = time.strftime("%H:%M:%S.", time.gmtime(self.time))   # !!! it's not proper timedelta (not in `time`)
        ss = "elapsed\n" + \
            f" clock time:                    {isotime}\n" + \
            f" process time:        {self.process_time}\n" + \
            f" performance counter: {self.perf_counter}\n"
        return ss

    def __add__(self, other):
        lt = self.time + other.time
        pt = self.process_time + other.process_time
        pc = self.perf_counter + other.perf_counter
        return DTimes(lt, pt, pc)

    def __sub__(self, other):
        lt = self.time - other.time
        pt = self.process_time - other.process_time
        pc = self.perf_counter - other.perf_counter
        return DTimes(lt, pt, pc)

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
