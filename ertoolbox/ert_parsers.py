import numpy as np
import pandas as pd
from pathlib import Path
import glob
import re
from typing import Union
from typing import List
import sys

from ertoolbox.ert_preprocessing import replace_electrode_positions
from ertoolbox.ert_preprocessing import calc_geometrical_factor
from ertoolbox.ert_preprocessing import TXRX_plot

from ertoolbox.ert_preprocessing import remove_measurements
from ertoolbox.ert_preprocessing import add_electrode_position
from ertoolbox.ert_preprocessing import remove_electrode_position

# from ert_mesh import mesh_confinedspace_2D
# from ert_mesh import mesh_halfspace_2D
# from ert_mesh import mesh_crosshole_2D
# from ert_mesh import mariosmesh

from ertoolbox.ert_inversion import prepare_inversion_datacontainer
from ertoolbox.ert_inversion import inversion
from ertoolbox.ert_inversion import export_to_dft

from ertoolbox.database_io import read_task

standard_colnames_electrodes = {
    "Cbl#": "Cable",
    "El#": "Electrode",
    "Elec-X": "Electrode_x",
    "Elec-Y": "Electrode_y",
    "Elec-Z": "Electrode_z",
    "Terrn-Z": "Terrain_z",
    "El.Num": "ElectrodeNumber",
}

standard_colnames_data = {
    "ID": "ID",
    "A": "A",
    "B": "B",
    "M": "M",
    "N": "N",
    "Appres": "Appres",
    "V/I,": "V/I",
    "Std.": "Std",
    "ContactR": "ContactR",
}


def colums_to_float(df, columns):
    for col in columns:
        df[col] = df[col].str.strip("+-").astype(float)
    return df


class ErtBase:
    """
    Class to parse my ERT files.
    Args:
        filepath (str): path to the file to be parsed
    """

    @property
    def data(self):
        return self._data

    @property
    def electrodes(self):
        return self._electrodes

    def __init__(self, filepath):
        self.__open_file(filepath)
        self.filepath = filepath

    def __open_file(self, filepath):
        if str(filepath).endswith("Data") or str(filepath).endswith("txt"):
            """
            For raw text data files (.data = MPT, .txt = ABEM)
            """
            with open(filepath, "r") as f:
                # self.lines = f.readlines()
                self.text = f.read()

        elif str(filepath).endswith(".db"):
            self.text = read_task(filepath, ids=(1, 1), includeIP=True)

    # Preprocessing
    def remove_measurements(
        self,
        elec=np.nan,
        ABMN=[np.nan, np.nan, np.nan, np.nan],
        index=np.nan,
    ):
        self._data = remove_measurements(self._data, elec, ABMN, index)

    def remove_electrode_position(
        self,
        electrode: Union[float, List[float]] = np.nan,
    ):
        self._electrodes = remove_electrode_position(self._electrodes, electrode)

    def add_electrode_position(self, Electrode, Electrode_x, Electrode_y, Electrode_z):
        self._electrodes = add_electrode_position(
            self._electrodes, Electrode, Electrode_x, Electrode_y, Electrode_z
        )

    def replace_electrode_positions(
        self, new_x=[np.nan, np.nan], new_y=[np.nan, np.nan], new_z=[np.nan, np.nan]
    ):
        self._electrodes = replace_electrode_positions(
            self._electrodes, new_x, new_y, new_z
        )

    def TXRX_plot(self, vmin=0, vmax=0):
        self._data = TXRX_plot(self._data, vmin, vmax)

    def calc_geometrical_factor(self):
        self._electrodes, self._data = calc_geometrical_factor(
            self._electrodes, self._data
        )

    # Meshing
    def mesh_confinedspace_2D(self, xmin, xmax, depth):
        mesh = mesh_confinedspace_2D(self._electrodes, xmin, xmax, depth)
        return mesh

    def mesh_halfspace_2D(self, xmin, xmax):
        mesh = mesh_halfspace_2D(self._electrodes, xmin, xmax)
        return mesh

    def mesh_crosshole_3D(self):
        mesh = mesh_crosshole_3D(self._electrodes)
        return mesh

    def mesh_crosshole_2D(self):
        mesh = mesh_crosshole_2D(self._electrodes)
        return mesh

    # Inversion -- fix inversion
    def prepare_inversion_datacontainer(self):
        data = prepare_inversion_datacontainer(self)
        return data

    def inversion(self, dc, mesh, saveresult, filename, folder):
        model = inversion(
            dc, mesh, saveresult=True, filename=filename, folder="C:/Users/leentvaa"
        )
        return model


class MPT(ErtBase):
    """
    Subclass for MPT system (.Data raw data files)

    Parameters
    ----------
    filepath : str

    sep : str, optional
        The separator used in the file. The default is "' '".
    ip : bool, optional
        Whether to calculate the IP from the data. The default is True.

    """

    def __init__(self, filepath, sep=" ", ip=True):
        super().__init__(filepath)
        self.sep = sep
        self.ip = ip  # later
        self.parse_file()

    def parse_file(self):
        self.split_file()
        self.parse_data()
        self.parse_electrodes()
        self.split_abmn()

    def split_file(self):
        self.find_data()
        self.find_electrodes()

    def find_electrodes(self):
        dstart = re.search(r"#elec_start\n", self.text).end()
        dend = re.search(r"#elec_end\n", self.text).start()
        self._electrodes = self.text[dstart:dend]

    def find_data(self):
        dstart = re.search(r"#data_start\n", self.text).end()
        dend = re.search(r"#data_end\n", self.text).start()
        self._data = self.text[dstart:dend]

    def parse_data(self):
        data = re.sub(r"[^\S\r\n]+", " ", self._data)
        data = data.splitlines()
        columns, units = data[:2]
        data = data[2:]
        wrong_cols = ["Gains", "Tx_V"]

        columns = [
            standard_colnames_data.get(col, col)
            for col in columns.lstrip("! ").split(self.sep)
            if col not in wrong_cols
        ]

        df = pd.DataFrame([d.rstrip(self.sep).split(self.sep) for d in data])
        df = df.iloc[:, : len(columns)]
        df.columns = columns
        outofrange = np.where(df == "out")[0]
        df = df.drop(outofrange)
        df = colums_to_float(df, ["Appres", "V/I", "ContactR"])
        self._data = df

    def parse_electrodes(self):
        electrodes = re.sub(r"[^\S\r\n]+", " ", self._electrodes)
        electrodes = electrodes.splitlines()
        columns = electrodes[0]
        wrong_cols = ["Type"]
        columns = [
            standard_colnames_electrodes.get(col, col)
            for col in columns.lstrip("! ").split(self.sep)
            if col not in wrong_cols
        ]

        df = pd.DataFrame(
            [
                d.rstrip(self.sep).replace(",", " ").split(self.sep)
                for d in electrodes[1:]
            ]
        )

        df = df.iloc[:, : len(columns)]
        df.columns = columns

        df = df[
            ["Electrode", "Electrode_x", "Electrode_y", "Electrode_z", "Terrain_z"]
        ].astype(float)

        self._electrodes = df

    def split_abmn(self):
        self._data["A"] = self._data["A"].apply(lambda x: float(x.split(",")[1]))
        self._data["B"] = self._data["B"].apply(lambda x: float(x.split(",")[1]))
        self._data["M"] = self._data["M"].apply(lambda x: float(x.split(",")[1]))
        self._data["N"] = self._data["N"].apply(lambda x: float(x.split(",")[1]))


class ABEM(ErtBase):
    """
    Subclass for ABEM system NOT FINISHED
    need to check how abem files look to write a reader for it.
    for .txt files abemn
    """

    def __init__(self, filepath, sep=" "):
        super().__init__(filepath)
        self.sep = sep
        self.parse_file()

    def parse_file(self):
        self.split_file()
        self.parse_all()
        self.parse_electrodes()
        self.parse_data()
        # self.split_abmn()

    def split_file(self):
        self.find_all()

    def find_all(self):
        dstart = re.search(r"N\tTime", self.text)
        dstart = dstart.span()[0]
        dend = re.search(r"-----", self.text)
        dend = dend.span()[0]
        self._all = self.text[dstart:dend]

    def parse_all(self):
        data = re.sub(r"[^\S\r\n]+", " ", self._all)
        data = data.splitlines()
        columns = data[:1]
        columns = [item.split() for item in columns][0]
        columns.insert(1, "Date")
        data = data[1:]
        df = pd.DataFrame([d.rstrip(self.sep).split(self.sep) for d in data])
        df.columns = columns
        # remove rows that are nan
        self._all = df

    def parse_electrodes(self):
        columns = [
            "Electrode",
            "Electrode_x",
            "Electrode_y",
            "Electrode_z",
            "Terrain_z",
        ]
        df = pd.DataFrame(columns=columns)
        df.columns = columns
        df["Electrode_x"] = pd.concat(
            [self._all["A(x)"], self._all["B(x)"], self._all["M(x)"], self._all["N(x)"]]
        )
        df["Electrode_y"] = pd.concat(
            [self._all["A(y)"], self._all["B(y)"], self._all["M(y)"], self._all["N(y)"]]
        )
        df["Electrode_z"] = pd.concat(
            [self._all["A(z)"], self._all["B(z)"], self._all["M(z)"], self._all["N(z)"]]
        )

        df = df[
            ~df.duplicated(
                subset=["Electrode_x", "Electrode_y", "Electrode_z"], keep="first"
            )
        ]

        df = df.reset_index(drop=True)
        df["Electrode"] = df.index + 1
        self._electrodes = df.astype(float)

    def parse_data(self):
        columns = [
            "ID",
            "A",
            "B",
            "M",
            "N",
            "Appres",
            "V/I",
            "Std",
            "Amp.",
            "Std",
            "Current",
            "ContactR",
            "Date_And_Time",
        ]
        df = pd.DataFrame(columns=columns)
        df.columns = columns
        df["Appres"] = self._all["App.R(Ohmm)"].astype(float)
        df["V/I"] = self._all["R(Ohm)"].astype(float)
        df["Std"] = self._all["N"]
        df["ID"] = self._all["N"]
        df["Amp."] = self._all["N"]
        df["Std"] = self._all["N"]
        df["Current"] = self._all["N"]
        df["ContactR"] = self._all["N"]
        df["Date_And_Time"] = self._all["Date"]

        # vraag aan bas
        ### ABMN ###
        for _, row in self._all.iterrows():
            if row["A(x)"] is not None:
                index = self._electrodes[
                    self._electrodes["Electrode_x"] == float(row["A(x)"])
                ].index
                if len(index > 0):
                    index2 = self._electrodes[
                        self._electrodes["Electrode_y"] == float(row["A(y)"])
                    ].index
                    index = index.intersection(index2)
                    if len(index > 0):
                        index3 = self._electrodes[
                            self._electrodes["Electrode_z"] == float(row["A(z)"])
                        ].index
                        index = index.intersection(index3)

                df["A"].iloc[_] = self._electrodes.iloc[index[0]]["Electrode"]
                index = []

            if row["B(x)"] is not None:
                index = self._electrodes[
                    self._electrodes["Electrode_x"] == float(row["B(x)"])
                ].index
                if len(index > 0):
                    index2 = self._electrodes[
                        self._electrodes["Electrode_y"] == float(row["B(y)"])
                    ].index
                    index = index.intersection(index2)
                    if len(index > 0):
                        index3 = self._electrodes[
                            self._electrodes["Electrode_z"] == float(row["B(z)"])
                        ].index
                        index = index.intersection(index3)

                df["B"].iloc[_] = self._electrodes.iloc[index[0]]["Electrode"]
                index = []

            if row["M(x)"] is not None:
                index = self._electrodes[
                    self._electrodes["Electrode_x"] == float(row["M(x)"])
                ].index
                if len(index > 0):
                    index2 = self._electrodes[
                        self._electrodes["Electrode_y"] == float(row["M(y)"])
                    ].index
                    index = index.intersection(index2)
                    if len(index > 0):
                        index3 = self._electrodes[
                            self._electrodes["Electrode_z"] == float(row["M(z)"])
                        ].index
                        index = index.intersection(index3)

                df["M"].iloc[_] = self._electrodes.iloc[index[0]]["Electrode"]
                index = []

            if row["N(x)"] is not None:
                index = self._electrodes[
                    self._electrodes["Electrode_x"] == float(row["N(x)"])
                ].index
                if len(index > 0):
                    index2 = self._electrodes[
                        self._electrodes["Electrode_y"] == float(row["N(y)"])
                    ].index
                    index = index.intersection(index2)
                    if len(index > 0):
                        index3 = self._electrodes[
                            self._electrodes["Electrode_z"] == float(row["N(z)"])
                        ].index
                        index = index.intersection(index3)

                df["N"].iloc[_] = self._electrodes.iloc[index[0]]["Electrode"]
                index = []

        df.dropna(subset=["A", "B", "M", "N"], inplace=True)
        self._data = df
