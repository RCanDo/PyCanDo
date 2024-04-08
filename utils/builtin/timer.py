#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
import time
from datetime import datetime


# %%
class TimeLapse:

    def __init__(self, t: float, unit: str = 's'):
        """
        t: float,
            timestamp, i.e. number of `unit`s from 1970-01-01 00:00:00 (POSIX);
        unit: str = 's'
            units in which `t` is given;
            currently only possible values are: 's' for seconds and 'n' for nanoseconds (1e-9 s);
        """
        self.unit = unit
        self.time = t

        if unit == 'n':
            seconds, nanoseconds = divmod(t, int(1e9))
        elif unit == 's':
            seconds = int(t)
            nanoseconds = (t - int(t)) * int(1e9)
        else:
            raise Exception("Currently only `units` 's' (seconds) and 'n' (nanoseconds) implemented.")

        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)

        self.days = days
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds
        self.nanoseconds = nanoseconds

    def __repr__(self):
        ss = f"TimeLapse({self.time}, '{self.unit}')\n" + \
             f"days: {self.days}, " + \
             f"hours: {self.hours}, minutes: {self.minutes}, seconds: {self.seconds}, nanoseconds: {self.nanoseconds}"
        return ss

    def __str__(self):
        days = f"{self.days} days " if self.days > 0 else ""
        ss = days + f"{self.hours:0>2}:{self.minutes:0>2}:{self.seconds:0>2}.{self.nanoseconds}"
        return ss


class SimpleTimer():
    """
    Example
    -------
    t0 = SimpleTimer()
    t0
    t0.time
    t0.time_start
    t0.isotime_start
    ...
    t0.stop()
    t0
    print(t0)
    t0.time_stop
    t0.isotime_stop
    """
    def __init__(self):
        self.time_start = time.time()       # float [s], precision 1e-9 [n]
        self.isotime_start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.time_start))
        self.stop(0)

    def stop(self, t: float = None):
        if t is None:
            self.time_stop = time.time()    # float [s], precision 1e-9 [n]
        else:
            self.time_stop = self.time_start + t
        self.isotime_stop = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.time_stop))
        self.diff = self.time_stop - self.time_start    # float [s], precision 1e-9 [n]
        self.time_diff = TimeLapse(self.diff, 's')
        self.__dict__ |= self.time_diff.__dict__

    def __repr__(self):
        ss = f"SimpleTimer()  from  {self.isotime_start}  to  {self.isotime_stop}\n" + \
             f"days: {self.days}, " + \
             f"hours: {self.hours}, minutes: {self.minutes}, seconds: {self.seconds}, nanoseconds: {self.nanoseconds}"
        return ss

    def __str__(self):
        ss = self.time_diff.__str__()
        return ss


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
        self.clock_time: float = time.time()
        self.process_time: float = time.process_time()
        self.perf_counter: float = time.perf_counter()

    def __sub__(self, other):
        dlt: float = self.clock_time - other.clock_time
        dpt: float = self.process_time - other.process_time
        dpc: float = self.perf_counter - other.perf_counter
        return DTimes(dlt, dpt, dpc)

    def __str__(self):
        isotime = time.strftime("%Y-%m-%d %H:%M:%S.", time.localtime(self.clock_time))
        ss = f"  local time:          {isotime}\n" + \
             f"  process time:        {self.process_time:>18_.9f} [s]\n" + \
             f"  performance counter: {self.perf_counter:>18_.9f} [s]\n"
        return ss

    def __repr__(self):
        return self.__str__()


class DTimes():
    """
    time delta for Times
    """
    def __init__(self, dlt: float, dpt: float, dpc: float):
        self.clock_time = TimeLapse(dlt)
        self.process_time = TimeLapse(dpt)
        self.perf_counter = TimeLapse(dpc)

    def __str__(self):
        ss = " elapsed" + \
             "\n  clock time:          " + self.clock_time.__str__() + \
             "\n  process time:        " + self.process_time.__str__() + \
             "\n  performance counter: " + self.perf_counter.__str__()
        return ss

    def __add__(self, other):
        lt = self.clock_time + other.clock_time
        pt = self.process_time + other.process_time
        pc = self.perf_counter + other.perf_counter
        return DTimes(lt, pt, pc)

    def __sub__(self, other):
        lt = self.clock_time - other.clock_time
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
    def __init__(self, title: str = None):
        self.title = title
        self.time_start = Times()
        self.stop(0)

    def stop(self, t: float = None):
        if t is None:
            self.time_stop = Times()
        elif t == 0:
            self.time_stop = self.time_start
        else:           # the same as None
            self.time_stop = Times()
        self.diff = self.time_stop - self.time_start
        self.elapsed = self.diff  # alias

    def __str__(self):
        ss = (self.title + "\n") if self.title is not None else ""
        ss += " start\n" + self.time_start.__str__() + \
              " stop\n" + self.time_stop.__str__() + \
              self.diff.__str__()
        return ss

    def __repr__(self):
        return self.__str__()

    def print(self, what: str = ""):
        if what == "":
            print(self.__str__())
        else:
            print(self.title)
            if what in ("diff", "elapsed"):
                ss = self.diff.__str__()
            elif what == "start":
                ss = self.time_start.__str__()
            elif what == "stop":
                ss = self.time_stop.__str__()
            else:
                raise Exception("`what` must be one of 'diff', 'elapsed', 'start', 'stop'.")
            print(ss)

