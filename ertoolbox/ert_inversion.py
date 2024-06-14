import numpy as np
import sys
import pandas as pd
import pygimli as pg
from pygimli.physics import ert
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import os


def export_to_vtk(self, filename, folder=None, size=(16, 10), **kwargs):
    path = Path(folder)
    m = pg.Mesh(self.paraDomain)
    m["Resistivity"] = self.paraModel(self.model)
    m["Resistivity (log10)"] = np.log10(m["Resistivity"])
    m["Coverage"] = self.coverage()
    m["S_Coverage"] = self.standardizedCoverage()
    m.exportVTK(os.path.join(path, str(filename)))
    return path


def prepare_inversion_datacontainer(data):
    dc = pg.DataContainerERT()

    data._electrodes = data.electrodes.reset_index(drop=True)
    for s in range(len(data.electrodes["Electrode_x"])):
        dc.createSensor(
            [
                data.electrodes["Electrode_x"][s],
                data.electrodes["Electrode_y"][s],
                data.electrodes["Electrode_z"][s],
            ]
        )

    for i in range(len(data._data["A"])):
        dc.addFourPointData(
            data.data["A"][i] - 1,
            data.data["B"][i] - 1,
            data.data["M"][i] - 1,
            data.data["N"][i] - 1,
        )

    dc.set("rhoa", (data.data["Appres"]))
    dc.set("k", data.data["GF"].tolist())
    dc.set("err", ert.estimateError(dc))

    return dc


def inversion(
    data,
    mesh,
    saveresult=False,
    cmin=1,
    cmax=100,
    filename="default",
    folder="C:/Users/leentvaa",
):
    mgr = ert.ERTManager(data, verbose=True)
    mgr.invert(mesh=mesh)
    mgr.showResult(cMin=cmin, cMax=cmax)
    # mgr.showResultAndFit()
    if saveresult == True:
        export_to_vtk(mgr, filename, folder)

    return mgr


def export_to_dft(mgr):
    raise NotImplementedError
    meshnew = pg.Mesh(mgr.paraDomain)

    cells_x = []
    cells_y = []
    cells_z = []
    resistivity = []

    for k in range(len(mgr.paraModel(mgr.model))):
        cells_x = np.append(cells_x, meshnew.cellCenter()[k][0])
        cells_y = np.append(cells_y, meshnew.cellCenter()[k][1])
        cells_z = np.append(cells_z, meshnew.cellCenter()[k][2])
        resistivity = np.append(resistivity, mgr.paraModel(mgr.model)[k])

    resistivity_list = []

    for j in range(len(resistivity)):
        location = data_input.Geometry(x=cells_x[j], y=cells_y[j], z=cells_z[j])
        resistivity_list.append(
            data_input.Data(
                location=location,
                variables=[
                    data_input.Variable(value=resistivity[j], label="resistivity")
                ],
            )
        )
    return resistivity_list
