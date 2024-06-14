import numpy as np
import pandas as pd
import pygimli as pg
import math
import matplotlib.pyplot as plt


def find_closest_value(arr, input_val):
    """
    Find closest value in array to input value

    Parameters
    ----------
    arr : array
        Array of values to search through
    input_val : float
        Value to find closest value to

    """
    closest_val = min(arr, key=lambda x: abs(x - input_val))
    return closest_val


def remove_electrode_position(electrodes, elec=[np.nan]):
    """
    Remove electrode positions from ERT data
    The electrode positions should be given in a numpy array. Place np.nan in the array if you want to skip a value.
    If you want to replace the entire array, give the new array as an argument. If you only want to replace a few values,
    give the new array as an argument and place np.nan in the array at the index where you want to keep the old values.

    Parameters
    ----------
    data : ERT data
        ERT data from ert_parsers
    electrode : array
        Array of electrode positions to remove

    """
    for el in range(len(elec)):
        electrodes.drop(
            electrodes[electrodes["Electrode"] == elec[el]].index,
            inplace=True,
        )

    return electrodes


def add_electrode_position(
    electrodes,
    Electrode,
    Electrode_x,
    Electrode_y,
    Electrode_z,
    Terrain_z=0,
    ElectrodeNumber=0,
    Cable="1",
):
    """
    Add electrode positions to ERT data

    Parameters
    ----------
    data : ERT data
        ERT data from ert_parsers
    Electrode : int
        Electrode number
    Electrode_x : float
        Electrode x position
    Electrode_y : float
        Electrode y position
    Electrode_z : float
        Electrode z position
    Terrain_z : float, optional
        Terrain z position. The default is 0.
    ElectrodeNumber : int, optional
        Electrode number. The default is 0.
    Cable : str, optional
        Cable number. The default is '1'.

    """
    ElectrodeNumber = Electrode

    newelec = pd.DataFrame(
        {
            "Cable": [Cable],
            "Electrode": [Electrode],
            "Electrode_x": [Electrode_x],
            "Electrode_y": [Electrode_y],
            "Electrode_z": [Electrode_z],
            "Terrain_z": [Terrain_z],
            "ElectrodeNumber": [ElectrodeNumber],
        }
    )
    print()
    cv = find_closest_value(electrodes["Electrode"], Electrode)
    loc = np.int(np.where(electrodes["Electrode"] == cv)[0]) + 1

    electrodes = pd.concat(
        [electrodes.iloc[:loc], newelec, electrodes.iloc[loc:]]
    ).reset_index(drop=True)
    return electrodes


def replace_electrode_positions(
    electrodes, new_x=[np.nan, np.nan], new_y=[np.nan, np.nan], new_z=[np.nan, np.nan]
):
    """
    Replace electrode positions
    The new electrode positions should be given in a numpy array. Place np.nan in the array if you want to skip a value.
    If you want to replace the entire array, give the new array as an argument. If you only want to replace a few values,
    give the new values as an argument and the rest should be set at np.nan.

    Parameters
    ----------
    data : ERT data
        ERT data from ert_parsers
    new_x : array, optional
        Array of new x positions. The default is [np.nan,np.nan].
    new_y : array, optional
        Array of new y positions. The default is [np.nan,np.nan].
    new_z : array, optional
        Array of new z positions. The default is [np.nan,np.nan].

    """

    for i in range(len(new_x)):
        if str(new_x[i]) == "nan":
            pass
        else:
            electrodes.loc[i, ("Electrode_x")] = new_x[i]

    for j in range(len(new_y)):
        if str(new_y[j]) == "nan":
            pass
        else:
            electrodes.loc[j, ("Electrode_y")] = new_y[j]

    for k in range(len(new_z)):
        if str(new_z[k]) == "nan":
            pass
        else:
            electrodes.loc[k, ("Electrode_z")] = new_z[k]

    return electrodes


def ABMN_electrode_distance(p1x, p1y, p1z, p2x, p2y, p2z):
    """
    Calculate distance between two electrodes

    Parameters
    ----------
    p1x : float
        x position of electrode 1
    p1y : float
        y position of electrode 1
    p1z : float
        z position of electrode 1
    p2x : float
        x position of electrode 2
    p2y : float
        y position of electrode 2
    p2z : float
        z position of electrode 2

    Returns
    -------
    distance : float
        Distance between two electrodes

    """
    p1x = np.array(p1x)
    p1y = np.array(p1y)
    p1z = np.array(p1z)
    p2x = np.array(p2x)
    p2y = np.array(p2y)
    p2z = np.array(p2z)

    distance = np.zeros(len(p1x))

    for i in range(len(p1x)):
        distance[i] = np.sqrt(
            np.power(p1x[i] - p2x[i], 2)
            + np.power(p1y[i] - p2y[i], 2)
            + np.power(p1z[i] - p2z[i], 2)
        )
    return distance


def calc_geometrical_factor(electrodes, data):
    """
    Calculate geometrical factor.
    """
    A = np.array(data["A"])
    for i in range(len(A)):
        A[i] = int(A[i]) - 1

    xcoords_a = electrodes.loc[A, ("Electrode_x")]
    ycoords_a = electrodes.loc[A, ("Electrode_y")]
    zcoords_a = electrodes.loc[A, ("Electrode_z")]

    B = np.array(data["B"])
    for i in range(len(B)):
        B[i] = int(B[i]) - 1

    xcoords_b = electrodes.loc[B, ("Electrode_x")]
    ycoords_b = electrodes.loc[B, ("Electrode_y")]
    zcoords_b = electrodes.loc[B, ("Electrode_z")]

    M = np.array(data["M"])
    for i in range(len(M)):
        M[i] = int(M[i]) - 1

    xcoords_m = electrodes.loc[M, ("Electrode_x")]
    ycoords_m = electrodes.loc[M, ("Electrode_y")]
    zcoords_m = electrodes.loc[M, ("Electrode_z")]

    N = np.array(data["N"])
    for i in range(len(N)):
        N[i] = int(N[i]) - 1

    xcoords_n = electrodes.loc[N, ("Electrode_x")]
    ycoords_n = electrodes.loc[N, ("Electrode_y")]
    zcoords_n = electrodes.loc[N, ("Electrode_z")]

    AM = ABMN_electrode_distance(
        xcoords_a, ycoords_a, zcoords_a, xcoords_m, ycoords_m, zcoords_m
    )
    AN = ABMN_electrode_distance(
        xcoords_a, ycoords_a, zcoords_a, xcoords_n, ycoords_n, zcoords_n
    )
    BM = ABMN_electrode_distance(
        xcoords_b, ycoords_b, zcoords_b, xcoords_m, ycoords_m, zcoords_m
    )
    BN = ABMN_electrode_distance(
        xcoords_b, ycoords_b, zcoords_b, xcoords_n, ycoords_n, zcoords_n
    )

    AM[AM > 0] = 1 / AM[AM > 0]
    AN[AN > 0] = 1 / AN[AN > 0]
    BM[BM > 0] = 1 / BM[BM > 0]
    BN[BN > 0] = 1 / BN[BN > 0]

    gf = AM - AN - BM + BN
    gf = 2 * np.pi / gf

    data["GF"] = gf
    data["Appres"] = gf * data["V/I"]

    return electrodes, data


def TXRX_plot(data, vmin=0, vmax=5000):
    fig, ax = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    ax[0, 0].scatter(
        data["A"].tolist(),
        data["M"].tolist(),
        c=data["Appres"].tolist(),
        vmin=vmin,
        vmax=vmax,
        cmap="rainbow",
    )
    ax[0, 0].set_title("A, M, Appres values")
    ax[0, 0].set_xlabel("A")
    ax[0, 0].set_ylabel("M")

    ax[1, 0].scatter(
        data["A"].tolist(),
        data["N"].tolist(),
        c=data["Appres"].tolist(),
        vmin=vmin,
        vmax=vmax,
        cmap="rainbow",
    )
    ax[1, 0].set_title("A, N, Appres values")
    ax[1, 0].set_xlabel("A")
    ax[1, 0].set_ylabel("N")

    ax[0, 1].scatter(
        data["B"].tolist(),
        data["M"].tolist(),
        c=data["Appres"].tolist(),
        vmin=vmin,
        vmax=vmax,
        cmap="rainbow",
    )
    ax[0, 1].set_title("B, M, Appres values")
    ax[0, 1].set_xlabel("B")
    ax[0, 1].set_ylabel("M")

    ax[1, 1].scatter(
        data["B"].tolist(),
        data["N"].tolist(),
        c=data["Appres"].tolist(),
        vmin=vmin,
        vmax=vmax,
        cmap="rainbow",
    )
    ax[1, 1].set_title("B, N, Appres values")
    ax[1, 1].set_xlabel("B")
    ax[1, 1].set_ylabel("N")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(
        ax[0, 0].scatter(
            data["A"].tolist(),
            data["M"].tolist(),
            c=data["Appres"].tolist(),
            vmin=vmin,
            vmax=vmax,
            cmap="rainbow",
        ),
        cax=cbar_ax,
    )

    plt.show()

    return data


def remove_measurements(
    data, elec=np.nan, ABMN=[np.nan, np.nan, np.nan, np.nan], index=np.nan
):
    """
    Remove data based on TXRX plot. This plot should be assessed by hand and a specialist should determine what data should be removed.
    You can remove data from entire electrodes and from ABMN combinations (so single measurements).
    we can add more stuff here like remove based on contact resistivity
    """
    if np.isnan(index) == False:
        data = data.drop(index)

    if np.isnan(elec) == False:
        removables = np.concatenate(
            (
                np.where(data["A"] == elec),
                np.where(data["B"] == elec),
                np.where(data["M"] == elec),
                np.where(data["N"] == elec),
            ),
            axis=1,
        )
        data = data.drop(np.unique(removables).tolist())
        data = data.reset_index(drop=True)

    if np.count_nonzero(np.isnan(ABMN)) == 3:
        if ABMN[0] != np.nan:
            removables = np.where(data["A"] == ABMN[0])
            data = data.drop(np.unique(removables).tolist())

        if ABMN[1] != np.nan:
            removables = np.where(data["B"] == ABMN[1])
            data = data.drop(np.unique(removables).tolist())

        if ABMN[2] != np.nan:
            removables = np.where(data["M"] == ABMN[2])
            data = data.drop(np.unique(removables).tolist())

        if ABMN[3] != np.nan:
            removables = np.where(data["N"] == ABMN[3])
            data = data.drop(np.unique(removables).tolist())

    if np.count_nonzero(np.isnan(ABMN)) == 2:
        print(ABMN, np.isnan(ABMN[1]))
        if np.isnan(ABMN[0]) == False and np.isnan(ABMN[1]) == False:
            removables = np.where((data["A"] == ABMN[0]) & (data["B"] == ABMN[1]))
            data = data.drop(np.unique(removables).tolist())

        if np.isnan(ABMN[0]) == False and np.isnan(ABMN[2]) == False:
            removables = np.where((data["A"] == ABMN[0]) & (data["M"] == ABMN[2]))
            data = data.drop(np.unique(removables).tolist())

        if np.isnan(ABMN[0]) == False and np.isnan(ABMN[3]) == False:
            removables = np.where((data["A"] == ABMN[0]) & (data["N"] == ABMN[3]))
            data = data.drop(np.unique(removables).tolist())

        if np.isnan(ABMN[1]) == False and np.isnan(ABMN[2]) == False:
            removables = np.where((data["B"] == ABMN[1]) & (data["M"] == ABMN[2]))
            data = data.drop(np.unique(removables).tolist())

        if np.isnan(ABMN[1]) == False and np.isnan(ABMN[3]) == False:
            removables = np.where((data["B"] == ABMN[1]) & (data["N"] == ABMN[3]))
            data = data.drop(np.unique(removables).tolist())

        if np.isnan(ABMN[2]) == False and np.isnan(ABMN[3]) == False:
            removables = np.where((data["M"] == ABMN[2]) & (data["N"] == ABMN[3]))
            data = data.drop(np.unique(removables).tolist())

    if np.count_nonzero(np.isnan(ABMN)) == 1:
        if (
            np.isnan(ABMN[0]) == False
            and np.isnan(ABMN[1]) == False
            and np.isnan(ABMN[2]) == False
        ):
            removables = np.where(
                (data["A"] == ABMN[0]) & (data["B"] == ABMN[1]) & (data["M"] == ABMN[2])
            )
            data = data.drop(np.unique(removables).tolist())

        if (
            np.isnan(ABMN[0]) == False
            and np.isnan(ABMN[2]) == False
            and np.isnan(ABMN[3]) == False
        ):
            removables = np.where(
                (data["A"] == ABMN[0]) & (data["M"] == ABMN[2]) & (data["N"] == ABMN[3])
            )
            data = data.drop(np.unique(removables).tolist())

        if (
            np.isnan(ABMN[0]) == False
            and np.isnan(ABMN[1]) == False
            and np.isnan(ABMN[3]) == False
        ):
            removables = np.where(
                (data["A"] == ABMN[0]) & (data["B"] == ABMN[1]) & (data["N"] == ABMN[3])
            )
            data = data.drop(np.unique(removables).tolist())

        if (
            np.isnan(ABMN[1]) == False
            and np.isnan(ABMN[2]) == False
            and np.isnan(ABMN[3]) == False
        ):
            removables = np.where(
                (data["B"] == ABMN[1]) & (data["M"] == ABMN[2]) & (data["N"] == ABMN[3])
            )
            data = data.drop(np.unique(removables).tolist())

    if np.count_nonzero(np.isnan(ABMN)) == 0:
        removables = np.where(
            (data["A"] == ABMN[0])
            & (data["B"] == ABMN[1])
            & (data["M"] == ABMN[2])
            & (data["N"] == ABMN[3])
        )
        data = data.drop(np.unique(removables).tolist())

    return data
