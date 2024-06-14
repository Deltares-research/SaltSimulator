import numpy as np
import pandas as pd
import pygimli as pg
from pygimli.physics import ert
import matplotlib.pyplot as plt
import glob


def simulate_mesh_forUG_leak(
    data,
    xmin,
    xmax,
    depth,
    setresloosesand,
    setresconfsand,
    setresleak,
    leakpos,
    leakr,
    addleak=True,
):
    # World
    world = pg.meshtools.createWorld(
        start=[xmin, -depth], end=[xmax, 0], worldMarker=True
    )

    layer1 = pg.meshtools.createRectangle(
        [0, -0.05], [7.75, -0.5], isclosed=False, marker=2, boundaryMarker=2
    )
    layer2 = pg.meshtools.createRectangle(
        [3.875, -0.5], [7.75, -1], isclosed=False, marker=2, boundaryMarker=2
    )

    bricks = pg.meshtools.createRectangle(
        start=[xmin, -0.05], end=[xmax, 0], islosed=False, marker=3, boundaryMarker=3
    )

    print("here")
    # Leak
    if leakpos == "Left":
        lr = (2, -1)
    elif leakpos == "Right":
        lr = (6, -1)

    leak = pg.meshtools.createCircle(
        pos=lr, radius=leakr, marker=4, boundaryMarker=4, area=0.01
    )

    # Combine
    world = world + layer1 + layer2 + bricks
    print("hi")
    # addregionmarkers for the leak if geometry crashes

    meshworld = pg.meshtools.createMesh(world, quality=1.2, area=0.01)
    meshcircle = pg.meshtools.createMesh(leak, quality=1.2, area=0.01)

    mesh = pg.meshtools.mergeMeshes([meshworld, meshcircle])

    # Resistivity
    kMap = [[1, setresconfsand], [2, setresloosesand], [4, setresleak], [3, 500]]
    K = pg.solver.parseMapToCellArray(kMap, mesh)

    # Plot
    pg.show(mesh, data=kMap, label=pg.unit("res"), showMesh=True)

    return mesh, K


def simulate_mesh_forUG(
    data,
    xmin,
    xmax,
    depth,
    setresloosesand,
    setresconfsand,
    setresleak,
    leakpos,
    leakr,
    addleak=True,
):
    # World
    world = pg.meshtools.createWorld(
        start=[xmin, -depth], end=[xmax, 0], worldMarker=True
    )

    layer1 = pg.meshtools.createRectangle(
        [0, -0.05], [7.75, -0.5], isclosed=False, marker=2, boundaryMarker=2
    )
    layer2 = pg.meshtools.createRectangle(
        [3.875, -0.5], [7.75, -1], isclosed=False, marker=2, boundaryMarker=2
    )

    bricks = pg.meshtools.createRectangle(
        start=[xmin, -0.05], end=[xmax, 0], islosed=False, marker=3, boundaryMarker=3
    )

    print("here")
    # Leak
    if leakpos == "Left":
        lr = (2, -1)
    elif leakpos == "Right":
        lr = (6, -1)

    leak = pg.meshtools.createCircle(
        pos=lr, radius=leakr, marker=4, boundaryMarker=4, area=0.01
    )

    # Combine
    world = world + layer1 + layer2 + bricks
    print("hi")
    # addregionmarkers for the leak if geometry crashes

    meshworld = pg.meshtools.createMesh(world, quality=1.2, area=0.01)

    mesh = meshworld

    # Resistivity
    kMap = [[1, setresconfsand], [2, setresloosesand], [4, setresleak], [3, 500]]
    K = pg.solver.parseMapToCellArray(kMap, mesh)

    # Plot
    pg.show(mesh, data=kMap, label=pg.unit("res"), showMesh=True)

    return mesh, K


def simulate_data_forUG(electrodes, data, all, mesh, K):
    from pygimli.physics import ert
    from ert_inversion import prepare_inversion_datacontainer

    elec = np.array(
        [
            electrodes["Electrode_x"],
            electrodes["Electrode_y"],
            electrodes["Electrode_z"],
        ]
    ).T

    dc = prepare_inversion_datacontainer(all)
    dc["a"] = dc["a"] - 64  # specific for this data set
    dc["b"] = dc["b"] - 64  # specific for this data set
    dc["m"] = dc["m"] - 64  # specific for this data set
    dc["n"] = dc["n"] - 64  # specific for this data set

    dd = ert.createData(elecs=elec, schemeName="dd")
    simdata_dd = simdata = ert.simulate(
        mesh=mesh, scheme=dd, res=K, noiseLevel=1, noiseAbs=1e-6, seed=1337
    )
    # pg.show(simdata_dd)
    # plt.title("Simulated data (dd)")
    simdata = ert.simulate(
        mesh=mesh, scheme=dc, res=K, noiseLevel=1, noiseAbs=1e-6, seed=1337
    )
    # pg.show(simdata)
    # plt.title("Simulated data (dc)")
    return simdata, simdata_dd


def simulate_data(EX, EY, EZ, mesh, K):
    from pygimli.physics import ert

    elec = np.array([EX, EY, EZ]).T
    dd = ert.createData(elecs=np.linspace(start=1, stop=40, num=21), schemeName="dd")

    simdata_dd = ert.simulate(
        mesh=mesh, scheme=dd, res=K, noiseLevel=1, noiseAbs=1e-6, seed=1337
    )

    return simdata, simdata_dd
