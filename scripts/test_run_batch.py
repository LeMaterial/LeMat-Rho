import json
import pytest
from unittest.mock import Mock

from run_utilities import run_batch, ID_SET

from pymatgen.core import Structure, Lattice

@pytest.fixture
def matpes_data():
    # Create a minimal, valid structure (e.g. cubic Fe)
    lattice = Lattice.cubic(3)
    species = ["Fe"]
    coords = [[0, 0, 0]]
    structure = Structure(lattice, species, coords)

    matpes_0 = {
        "matpes_id": "test_id_0",
        "structure": structure.as_dict(),
    }
    matpes_1 = {
        "matpes_id": "test_id_1",
        "structure": structure.as_dict(),
    }
    metadata_0 = {"mat_id": "test_id_0", "compare with": "matpes-0", "flow": "off_equilibrium"}
    metadata_1 = {"mat_id": "test_id_1", "compare with": "matpes-1", "flow": "off_equilibrium"}
    batch_metadata = {
        "batch_name": "compare_matpes",
        "description": "Compare MatPES data",
        "flow": "off_equilibrium",
    }
    structure_list = [
        Structure.from_dict(matpes_0["structure"]),
        Structure.from_dict(matpes_1["structure"]),
    ]
    return [metadata_0, metadata_1], structure_list, batch_metadata


def test_run_batch_no_duplicate_submission(matpes_data):
    metadata_list, structure_list, batch_metadata = matpes_data

    # Clear global ID_SET for a clean test run!
    ID_SET.clear()

    mock_calc_func = Mock()

    # First batch: should call for both jobs
    run_batch(
        metadata_list=metadata_list,
        structure_list=structure_list,
        batch_metadata=batch_metadata,
        calc_func=mock_calc_func,
        ncore=1,
        kpar=1,
        ignore_memory_check=True,
        worker="test_function_worker",
        nbands=[1, 1],
        nkpts=[1, 1],
    )

    assert mock_calc_func.call_count == 2

    # Second batch: should skip all, so NOT called again
    mock_calc_func.reset_mock()
    run_batch(
        metadata_list=metadata_list,
        structure_list=structure_list,
        batch_metadata=batch_metadata,
        calc_func=mock_calc_func,
        ncore=1,
        kpar=1,
        ignore_memory_check=True,
        worker="test_function_worker",
        nbands=[1, 1],
        nkpts=[1, 1],
    )
    assert mock_calc_func.call_count == 0
