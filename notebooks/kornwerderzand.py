# Global method
import numpy as np
import pandas as pd
import pygimli as pg
from pathlib import Path

import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.interpolate import Rbf
from scipy.spatial import Delaunay, cKDTree

from pygimli.physics import ert

# Local method
from ertoolbox import inversion
from ertoolbox import ert_postprocessing

directory = Path(
    "p:/11209233-mad09a2023ijsselmeer3d/C_Work/02_FM/03_postprocessing/09_input_tomografische_metingen/voorhaven_kwz/sal_tem/"
)

# Loop over all CSV files in the specified directory
for csv in directory.glob("*.csv"):
    data = pd.read_csv(csv)

    data.columns = [
        "x",
        "z",
        "salinity",
        "temperature",
    ]  # for now these columns are nesseccary!!

    data["resistivity"] = 1 / ert_postprocessing.salinity_to_conductivity(
        data["salinity"], data["temperature"]
    )
    print(data["salinity"])

    # Define grid parameters
    x_min, x_max = data["x"].min(), data["x"].max()
    z_min, z_max = data["z"].min(), data["z"].max()
    grid_size = 50  # Number of grid points along each axis

    # Create a regular grid
    x_grid = np.linspace(x_min, x_max, grid_size)
    z_grid = np.linspace(z_min, z_max, grid_size)
    X_grid, Z_grid = np.meshgrid(x_grid, z_grid)
    Y_grid = np.zeros(np.shape(X_grid))

    grid_points = np.column_stack([X_grid.flatten(), Z_grid.flatten()])

    # Prepare data for interpolation
    points = data[["x", "z"]].values
    values = data["resistivity"].values

    # Interpolate
    grid_values = griddata(points, values, (X_grid, Z_grid), method="linear")

    # Create 2D Pygimli  mesh
    mesh = pg.Mesh(2)

    cs = []
    cr = []

    # Make nodes for all datapoints
    for p in grid_points:
        c = mesh.createNode((p[0], p[1]))
        cs.append(c)

    # Triangulate points into mesh with triangles
    tri = Delaunay(grid_points)
    triangles = tri.simplices

    # Calculate centroids of the triangles
    centroids = np.array(
        [
            np.mean(grid_points[tri.simplices[i]], axis=0)
            for i in range(len(tri.simplices))
        ]
    )

    # Find closest grid point to the triangle
    tree = cKDTree(grid_points)
    distances, indices = tree.query(centroids)

    # Add value to the grid cell
    triangle_values = grid_values.flatten()[indices]

    # Add triangles to mesh object
    tlist = []
    for t in range(0, len(triangles)):
        mesh.createTriangle(
            cs[triangles[t][0]], cs[triangles[t][1]], cs[triangles[t][2]], marker=t
        )
        tlist = np.append(tlist, t)

    # Add boundaries
    mesh.createNeighborInfos()

    # Create a DataFrame
    df = pd.DataFrame(
        {"x": X_grid.flatten(), "z": Z_grid.flatten(), "value": grid_values.flatten()}
    )

    # Only use valid data points
    df = df.dropna()

    # #take 1 node above the bottom
    # min_z_indices = df.groupby('x')['z'].idxmin()
    # df = df.drop(min_z_indices)
    # df.reset_index(drop=True, inplace=True)

    # Find the bottom
    min_z_indices = df.groupby("x")["z"].idxmin()
    min_z_df = df.loc[min_z_indices]
    min_z_df.reset_index(drop=True, inplace=True)

    # Highlight the points where z is minimum for each unique x
    plt.scatter(min_z_df["x"], min_z_df["z"], color="red", s=100, label="Min Z per X")

    # Build electrode array
    EX = min_z_df["x"]
    EZ = min_z_df["z"]

    electrodes = [[x, y] for x, y in zip(EX, EZ)]

    # Create measurement scheme
    scheme_dd = ert.createData(elecs=electrodes, schemeName="dd")

    # Make simulation data
    simdata = ert.simulate(
        mesh=mesh,
        scheme=scheme_dd,
        res=triangle_values,
        noiseLevel=1,
        noiseAbs=1e-6,
        seed=1337,
    )

    simdata.remove(simdata["rhoa"] < 0)

    # Inverse for check
    inversion_dd_1 = inversion(
        simdata, mesh, saveresult=True, filename=str(csv), folder="../Kornwerderzand/"
    )
    break
