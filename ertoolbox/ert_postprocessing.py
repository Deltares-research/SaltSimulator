import numpy as np
import pandas as pd
import pygimli as pg
import math
import matplotlib.pyplot as plt

# not yet implemented in parser object!!!


def conductivity_to_salinity(conductivity, temp):
    """
    Convert conductivity and corresponding temperature value to salinity
    1-value only functioin

    Parameters
    ----------
    conductivity : float
        Conductivity [mS/m]
    temp : float
        Temperature [Celcius]

    Returns
    -------
    salinity : float
        Salinity value in [ ]

    """

    mt = 1 / (np.power(0.008018 * temp + 1.0609, 2) - 0.5911)
    salinity = np.power(conductivity * mt / 2.134, 1 / 0.92)

    return salinity
