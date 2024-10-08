import numpy as np
import pandas as pd
import pygimli as pg
import math
import matplotlib.pyplot as plt

# not yet implemented in parser object!!!


def conductivity_to_salinity(conductivity, temperature):
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

    mt = 1 / (np.power(0.008018 * temperature + 1.0609, 2) - 0.5911)
    salinity = np.power(conductivity * mt / 2.134, 1 / 0.92)

    return salinity


def salinity_to_conductivity(salinity, temperature):
    """
    Convert salinity and corresponding temperature value to conductivity
    1-value only functioin

    Parameters
    ----------
    salinity : float
        Salinity value in [ ]

    temp : float
        Temperature [Celcius]

    Returns
    -------
    conductivity : float
        Conductivity [mS/m]

    """
    mt = 1 / (np.power(0.008018 * temperature + 1.0609, 2) - 0.5911)
    conductivity = (salinity**0.92 * 2.134) / mt

    return conductivity
