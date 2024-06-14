import numpy as np
import pandas as pd
import pygimli as pg
from pygimli.physics import ert
import matplotlib.pyplot as plt
import glob


def mesh_halfspace_3D(data):
    raise NotImplementedError("not tested")
    """
    Generate mesh for a halfspace situation (field aquisition, no physical boundaries)
    automatically make the halfspace length of the electrode array + 2x this length on each side. 
    Make an option later to change this manually

    
    """
    xmin, xmax = min(data._electrodes["Electrode_x"]), max(
        data._electrodes["Electrode_x"]
    )
    ymin, ymax = min(data._electrodes["Electrode_y"]), max(
        data._electrodes["Electrode_y"]
    )
    # zmin, zmax = min(data._electrodes['Electrode_z']), max(data._electrodes['Electrode_z'])

    # think of something that makes the depth of the mesh automatically correct or smth.
    depth = -(2 * xmax)

    # not correct yet
    world = pg.meshtools.createWorld(
        start=[-(xmax - xmin) * 2, -(ymax - ymin) * 2, 0],
        end=[(xmax - xmin) * 2, (ymax - ymin) * 2, depth],
    )

    # polygon = pg.meshtools.createPolygon()

    box = pg.meshtools.createCube(
        size=[xmax - xmin, ymax - ymin, abs(depth)], start=[xmin, ymin, 0]
    )

    halfspace = pg.meshtools.createWorld(
        start=[-(xmax - xmin) * 2, -(ymax - ymin) * 2],
        end=[(xmax - xmin) * 2, (xmax - xmin) * 2],
        boundaryMarker=1,
    )
    focus = pg.meshtools.createCube(size=[xmax - xmin, ymax - ymin], start=[xmin, ymin])
    geometry = halfspace + focus
    square = pg.meshtools.createMesh(geometry)
    box = pg.meshtools.extrudeMesh(
        square, a=-(np.geomspace(1, depth + 1, depth + 1) - 1)
    )
    pg.show(box)


def mesh_confinedspace_3D(data, xsize, ysize, depth):
    raise NotImplementedError("not tested")
    geometry = pg.meshtools.createWorld(
        start=[0, 0], end=[xsize, ysize], boundaryMarker=1
    )
    square = pg.meshtools.createMesh(geometry)
    box = pg.meshtools.extrudeMesh(
        square, a=-(np.geomspace(1, depth + 1, depth + 1) - 1)
    )

    # Refinement
    for po in range(len(data._electrodes["Electrode_x"])):
        box.createNode(
            [
                data._electrodes.loc[po, ("Electrode_x")],
                data._electrodes.loc[po, ("Electrode_y")],
                data._electrodes.loc[po, ("Electrode_z")],
            ],
            marker=-99,
        )
        box.createNode(
            [
                data._electrodes.loc[po, ("Electrode_x")],
                data._electrodes.loc[po, ("Electrode_y")],
                data._electrodes.loc[po, ("Electrode_z")] - 0.05,
            ]
        )
        box.createNode(
            [
                data._electrodes.loc[po, ("Electrode_x")] + 0.05,
                data._electrodes.loc[po, ("Electrode_y")],
                data._electrodes.loc[po, ("Electrode_z")],
            ]
        )
        box.createNode(
            [
                data._electrodes.loc[po, ("Electrode_x")] - 0.05,
                data._electrodes.loc[po, ("Electrode_y")],
                data._electrodes.loc[po, ("Electrode_z")],
            ]
        )
        box.createNode(
            [
                data._electrodes.loc[po, ("Electrode_x")],
                data._electrodes.loc[po, ("Electrode_y")] + 0.05,
                data._electrodes.loc[po, ("Electrode_z")],
            ]
        )
        box.createNode(
            [
                data._electrodes.loc[po, ("Electrode_x")],
                data._electrodes.loc[po, ("Electrode_y")] - 0.05,
                data._electrodes.loc[po, ("Electrode_z")],
            ]
        )

    # Pivot electrodes
    box.createNode([xsize / 2, ysize / 2, -depth / 2], marker=-999)
    # Refinmemt
    box.createNode([xsize / 2, ysize / 2, -depth / 2 - 0.05], marker=-999)

    # Calibration nog checken
    box.createNode([xsize / 4, ysize / 4, 0.0], marker=-1000)
    # Refinemnet
    box.createNode([xsize / 4, ysize / 4, -0.05], marker=-1000)
    pg.show(box)


def mesh_confinedspace_2D(electrodes, xmin, xmax, depth):
    box = pg.meshtools.createWorld(start=[xmin, -depth], end=[xmax, 0])
    xsize = abs(xmax - xmin)

    electrodes = electrodes.reset_index(drop=True)

    # Refinement
    for po in range(len(electrodes["Electrode_x"])):
        box.createNode(
            [
                electrodes.loc[po, ("Electrode_x")],
                electrodes.loc[po, ("Electrode_z")],
            ],
            marker=-99,
        )
        box.createNode(
            [
                electrodes.loc[po, ("Electrode_x")],
                electrodes.loc[po, ("Electrode_z")] - 0.05,
            ]
        )

    # Pivot electrodes
    box.createNode([xsize / 2, -depth / 2], marker=-999)
    # Refinment
    box.createNode([xsize / 2, -depth / 2 - 0.05], marker=-999)

    # Calibration nog checken
    box.createNode([xsize / 4, 0.0], marker=-1000)
    # Refinement
    box.createNode([xsize / 4, -0.05], marker=-1000)

    mesh = pg.meshtools.createMesh(box, quality=1.2, area=0.005)
    # pg.show(mesh)
    return mesh


def mesh_halfspace_2D(electrodes, xmin, xmax,xbound= abs(xmax - xmin), ybound=abs(ymax - ymin)):
    depth = 0.5 * (xmax - xmin)
    box = pg.meshtools.createWorld(start=[xmin, -depth], end=[xmax, 0])
    xsize = abs(xmax - xmin)
    electrodes = electrodes.reset_index(drop=True)

    # Refinement
    for po in range(len(electrodes["Electrode_x"])):
        box.createNode(
            [
                electrodes.loc[po, ("Electrode_x")],
                electrodes.loc[po, ("Electrode_z")],
            ],
            marker=-99,
        )
        box.createNode(
            [
                electrodes.loc[po, ("Electrode_x")],
                electrodes.loc[po, ("Electrode_z")] - 0.05,
            ]
        )

    # Pivot electrodes
    box.createNode([xsize / 2, -depth / 2], marker=-999)
    # Refinment
    box.createNode([xsize / 2, -depth / 2 - 0.05], marker=-999)

    # Calibration nog checken
    box.createNode([xsize / 4, 0.0], marker=-1000)
    # Refinement
    box.createNode([xsize / 4, -0.05], marker=-1000)
    #
    # box = pg.meshtools.appendTriangleBoundary(box, marker=1, xbound=5, ybound=5)
    mesh = pg.meshtools.createMesh(box, quality=1.2, area=0.005)
    mesh = pg.meshtools.appendTriangleBoundary(mesh, marker=1, xbound=xbound, ybound=ybound)
    return mesh


def mesh_crosshole_2D(data):
    ex = np.unique(data["Electrode_x"])
    ez = np.unique(data["Electrode_z"])
    dx = 0.3
    nb = 8
    xmin, xmax = min(ex) - nb * dx, max(ex) + nb * dx
    zmin, zmax = min(ez) - nb * dx, 0
    x = np.arange(xmin, xmax + 0.001, dx)
    z = np.arange(zmin, zmax + 0.001, dx)
    z[-1] = 0

    grid = pg.meshtools.createGrid(x, z, marker=2)
    ax, cb = pg.show(grid)
    ax.plot(data["Electrode_x"], data["Electrode_z"], "mx")
    print(grid)
    mesh = pg.meshtools.appendTriangleBoundary(
        grid, marker=1, boundary=5, worldMarkers=1
    )
    pg.show(mesh, markers=True)

    return mesh


def mesh_crosshole_3D(data):
    # Pieter
    # ex = np.unique(data["Electrode_x"])
    # ez = np.unique(data["Electrode_y"])
    # dx = 0.25
    # nb = 2
    # xmin, xmax = min(ex) - nb * dx, max(ex) + nb * dx
    # zmin, zmax = min(ez) - nb * dx, 0
    # x = np.arange(xmin, xmax + 0.001, dx)
    # z = np.arange(zmin, zmax + 0.001, dx)
    # grid = pg.meshtools.createGrid(x, z, marker=2)
    # mesh = pg.meshtools.appendTriangleBoundary(
    #     grid, marker=1, xbound=25, ybound=25, worldMarkers=1
    # )

    elPosXY = np.unique(
        np.column_stack([data["Electrode_x"], data["Electrode_y"]]), axis=0
    )
    print(elPosXY)
    rect = pg.meshtools.createRectangle(pnts=elPosXY, minBBOffset=1.4, marker=2)
    for elpos in elPosXY:
        rect.createNode(*elpos, 0)

    ax, cb = pg.show(rect)
    _ = ax.plot(*elPosXY.T, "mx")

    bnd = 5
    rectMesh = pg.meshtools.createMesh(rect, quality=4, area=1.5)
    print(rectMesh)
    mesh2d = pg.meshtools.appendTriangleBoundary(
        rectMesh, boundary=bnd, isSubSurface=False, marker=1
    )
    ax, cb = pg.show(mesh2d, markers=True, showMesh=True)
    _ = ax.plot(*elPosXY.T, "mx")

    dTop, dBot = 3.5, 10.7
    # dzIn, dzOut = 0.3, 0.7
    # zTop = -np.arange(0, dTop, dzOut)  # the upper layer
    # zMid = -np.arange(dTop, dBot, dzIn)  # the middle
    # zBot = -np.arange(dBot, dBot + bnd + 0.1, dzOut)  # the lower layer
    # zVec = np.concatenate([zTop, zMid, zBot])  # all vectors together
    zVec = data["Electrode_z"].tolist()
    print(zVec)
    mesh = pg.meshtools.createMesh3D(
        mesh2d, zVec, pg.core.MARKER_BOUND_HOMOGEN_NEUMANN, pg.core.MARKER_BOUND_MIXED
    )
    print(mesh)
    for c in mesh.cells():
        cd = -c.center().z()  # center depth
        if cd < dTop or cd > dBot:
            c.setMarker(1)

    mesh["region"] = pg.Vector(mesh.cellMarkers())
    sli = pg.meshtools.extract2dSlice(mesh)
    ax, cb = pg.show(sli, "region", showMesh=True)
    _ = ax.plot(data["Electrode_x"], data["Electrode_y"], "mo", markersize=1)
    return mesh


def mariosmesh(pos, invbound=2, bound=10):
    import pygimli.meshtools as mt

    """Generate mesh around electrodes."""
    xmin, xmax = min(pg.x(pos)), max(pg.x(pos))
    ymin, ymax = min(pg.z(pos)), max(pg.z(pos))
    print(xmin, xmax, ymin, ymax)
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    world = mt.createWorld(start=[0, -2.0], end=[7.75, 0])
    # world.translate([xmid, ymid, 0])  # some bug in createWorld!

    maxdep = min(pg.z(pos)) - invbound
    sx = xmax - xmin + invbound * 2
    sy = ymax - ymin + invbound * 2

    # box=mt.createRectangle(start=[xmin-1, ymin], end=[xmax+1, 0],
    #                       marker=2,area=0.1)
    verts = [[xmax, 0], [xmin, 0]]
    temp = np.array(pos)

    for i in range(0, len(temp)):
        verts.append([temp[i, 0], temp[i, 2] - 0.1])
    box = mt.createPolygon(verts, isClosed=True, marker=2, area=0.005)
    # box=mt.createRectangle(start=[xmin-1, ymin], end=[xmax+1, 0],
    #                       marker=2,area=0.1)
    pos_xy = np.array(pos)
    pos_xy = np.c_[pos_xy[:, 0], pos_xy[:, 1]]
    geom = world
    for po in pos_xy:
        geom.createNode(po, marker=-99)
        geom.createNode([po[0], po[1] - 0.005], marker=-99)

        # geom.createNode(po - pg.Pos(0, 0.005))  # refinement

    # geom.exportPLC("mesh.poly")  # tetgen
    # geom.exportVTK("geom.vtk")  # vtk
    mesh = mt.createMesh(geom, quality=1.2, area=0.005)
    return mesh
