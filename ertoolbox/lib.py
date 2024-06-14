import sqlite3
import numpy as np
import pandas as pd

from contextlib import contextmanager

import time
from functools import wraps

@contextmanager
def db_connect(name):
    """
    Context manager for database connections.

    :param name:
    :return:
    """
    connection = sqlite3.connect(name)
    print('Connected!')
    yield connection
    connection.close()
    print('Connection closed!')

def my_timer(func):
    @wraps(func)
    def decorated(*args,**kwargs):
        # BEFORE
        start = time.time()
        # CALL
        wrapped = func(*args,**kwargs)
        # AFTER
        end = time.time()
        print("Time elapsed {}".format(end-start))
        return wrapped
    return decorated


def geometric_factor(a: float, b: float, m: float, n: float):
    """ 
    2-D geometric factor for surface ERT measurement
    
    :param a: x-position of current electrode A
    :param b: x-position of current electrode B
    :param m: x-position of potential electrode M
    :param n: x-position of potential electrode N
    :return: The 2-D geometric factor of the given electrode configuration
    :rtype: float
    """
    am = np.abs(a-m)
    an = np.abs(a-n)
    bm = np.abs(b-m)
    bn = np.abs(b-n)
    return 2*np.pi / ( (1/am)-(1/an)-(1/bm)+(1/bn))

def focus_point(a: float, b: float, m: float, n: float): # -> tuple[float, float]:
    """ Focus point of a multi-gradient array
    
    :param a: x-position of current electrode A
    :param b: x-position of current electrode B
    :param m: x-position of potential electrode M
    :param n: x-position of potential electrode N
    :return: A tuple with the focus point for the given configuration (Fx, Fz)
    :rtype: tuple(float, float)"""
    xmn = (m+n) / 2
    z = min( [ (xmn-a), (b-xmn)] ) / 3
    return (xmn, z)