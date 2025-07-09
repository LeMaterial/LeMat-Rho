import atomate2  # noqa: F401
from jobflow import SETTINGS  # noqa: F401
from jobflow.core.store import JobStore
from fireworks import LaunchPad
import pandas as pd  # noqa: F401
from pymatgen.core import Structure
import numpy as np  # noqa: F401
from typing import Literal, List, Dict, Callable, Any, Optional
from run_calculation import (
    relax,
    relax_start_pbe,
    static_calculation,
    static_calculation_off_equilibrium,
)

"""Utility functions for running calculations in batches and managing job submissions.
The module loads the submitted mat_ids from the LaunchPad and provides functions to run calculations in batches.
The mat_ids should be added as a user index to the launchpad to ensure they can be queried efficiently.
When we know more about the setup we can think of a better solution for the ID_SET and we can implement some heuristics for NCORE, KPAR, nbands, and nkpts.
"""
def get_import_string(func):
    return func.__module__ + '.' + func.__name__

def get_submitted_ids() -> set:
    """
    Get a set of submitted mat_ids for jobs with a mat_id in the Firework spec.
    Returns:
        set: Set of submitted mat_id strings.
    """
    lpad = LaunchPad.from_file("/home/sjonathan/atomate/config/lematrho_launchpad.yaml")
    query_filter = {"spec.mat_id": {"$exists": True}}
    return_fields = {"spec.mat_id": 1, "_id": 0}
    results = lpad.fireworks.find(query_filter, return_fields)
    mat_ids = {result["spec"]["mat_id"] for result in results if "spec" in result and "mat_id" in result["spec"]}
    return mat_ids

ID_SET = get_submitted_ids()
print(f"Found {len(ID_SET)} submitted mat_ids in LaunchPad.")

def get_mat_ids_successful_jobs(
    store: JobStore,
    calc_type: Literal[
        "LeMatRhoStaticMaker-matpes",
        "LeMatRhoStaticMaker",
        "LeMatRhoPreStaticMaker",
        "LeMatRhoRelaxMaker",
    ] = "LeMatRhoStaticMaker",
) -> List[str]:
    """
    Get a list of successful mat_ids from the JobStore filtered by a single calc_type.
    Args:
        store (JobStore): The JobStore instance to query.
        calc_type (str): The calculation type to filter by.
    Returns:
        List[str]: List of successful mat_ids.
    """
    query_filter = {
        "metadata.mat_id": {"$exists": True},
        "name": {"$regex": calc_type},
    }
    docs = list(store.query(query_filter, properties=["metadata", "name"]))
    mat_ids = [doc["metadata"]["mat_id"] for doc in docs]
    return mat_ids

def run_batch(
    metadata_list: List[dict],
    structure_list: List[Structure],
    batch_metadata: Dict[str, Any],
    calc_func: Callable,
    ncore: int,
    kpar: int,
    nbands: List[int],
    nkpts: List[int],
    ignore_memory_check: bool = False,
    worker: str = None,
    launchpad_path: Optional[str] = None,
) -> None:
    """
    Run a batch of calculations.

    Args:
        metadata_list (List[dict]): List of metadata dictionaries for each calculation.
        structure_list (List[Structure]): List of structure objects corresponding to each calculation.
        batch_metadata (Dict[str, Any]): Additional metadata for the batch.
        calc_func (Callable): The calculation function to run.
        ncore (int): Number of cores per calculation.
        kpar (int): K-point parallelization parameter.
        nbands (List[int]): List of nbands values for the batch.
        nkpts (List[int]): List of nkpts values for the batch.
        ignore_memory_check (bool, optional): If True, skip the memory check. Default is False.
        worker (str, optional): Worker identifier for the workflow manager.
        launchpad_path (Optional[str], optional): Path to the LaunchPad database. If not provided, it will use the default settings.

    Returns:
        None
    """
    if (
        (max(nbands) - min(nbands) > 24)
        or ((max(nkpts) - min(nkpts)) / max(nkpts) > 0.1)
    ) and not ignore_memory_check:
        raise ValueError(
            "The memory demands of the calculations are not roughly equal. "
            "Please check the nbands and nkpts values."
        )

    global ID_SET  # In case you want to update the global set after each submission

    for struct, metadata in zip(structure_list, metadata_list):
        mat_id = metadata.get("mat_id")
        if not mat_id:
            continue
        if mat_id not in ID_SET:
            metadata["batch_metadata"] = batch_metadata
            metadata["calc_func"] = get_import_string(calc_func)
            calc_func(
                struct,
                metadata=metadata,
                NCORE=ncore,
                KPAR=kpar,
                worker=worker,
                launchpad_path=launchpad_path,
            )
            ID_SET.add(mat_id)
        else:
            print(f"Skipping {mat_id} as it is already in ID_SET.")