PACKAGE_NAME = "discov_ert"
VERSION = "1.0.0"

from ert_parsers import MPT

from ert_preprocessing import replace_electrode_positions
from ert_preprocessing import calc_geometrical_factor
from ert_preprocessing import TXRX_plot
from ert_preprocessing import remove_measurements
from ert_preprocessing import remove_electrode_position
from ert_preprocessing import add_electrode_position

from ert_mesh import mesh_confinedspace_2D
from ert_mesh import mesh_halfspace_2D
from ert_mesh import mariosmesh

from ert_inversion import prepare_inversion_datacontainer
from ert_inversion import inversion

sys.path.append("~/Reader_Aris")
from database_io import read_task
