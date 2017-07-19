from pathlib import Path
from time import strftime

import numpy as np


def timestamp() -> str:
    return strftime('%Y%m%d-%H%M%S')


def create_timestamp_directory():
    p = Path(".") / "plots" / timestamp()

    p.mkdir(exist_ok=True)

    return p


timestamp_directory = create_timestamp_directory()

from timeit import default_timer as timer

all_t_so_far = []


class benchmark(object):
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *args):
        t = timer() - self.start

        all_t_so_far.append(t)

        self.time = t

        print(f'{self.message} current {t}s, average {np.average(all_t_so_far)}s +- {np.std(all_t_so_far)}')
