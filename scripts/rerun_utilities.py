import importlib
import os
import shutil
from fireworks import Firework, LaunchPad
from pymatgen.core import Structure
from jobflow import OutputReference
from jobflow.core.store import JobStore
from typing import Literal, List, Dict, Callable, Any, Optional, Tuple

def get_func_from_string(path: str) -> Callable:
    """
    Given a string like 'module.submodule.func', import and return the function.
    """
    module_path, func_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def mpi_tasks_from_worker(worker: str) -> Optional[int]:
    """
    Extracts the MPI tasks from a worker string.
    """
    try:
        return int(worker.split("_")[-1])
    except ValueError:
        print(f"Could not extract MPI tasks from worker string: {worker}")
        return None

def get_data_from_fw(fw_doc: dict, store: JobStore) -> Tuple[Structure, str, int, int, dict, Optional[int], Callable]:
    """
    Extracts the structure, worker, KPAR, and metadata from a Firework document.
    """
    if len(fw_doc["spec"]["_tasks"][0]["job"]["function_args"]) > 0:
        struct = Structure.from_dict(fw_doc["spec"]["_tasks"][0]["job"]["function_args"][0])
    else:
        struct_ref = fw_doc["spec"]["_tasks"][0]["job"]["function_kwargs"]["structure"]
        reference = OutputReference.from_dict(struct_ref)
        struct = reference.resolve(store)
    metadata = fw_doc["spec"]["_tasks"][0]["job"]["metadata"]
    worker = fw_doc["spec"]["_fworker"]
    KPAR = fw_doc["spec"]["_tasks"][0]["job"]["function"]["@bound"]["input_set_generator"]["user_incar_settings"]["KPAR"]
    NCORE = fw_doc["spec"]["_tasks"][0]["job"]["function"]["@bound"]["input_set_generator"]["user_incar_settings"]["NCORE"]
    mpi_tasks = mpi_tasks_from_worker(worker)
    calc_func_str = metadata.get("calc_func", "run_calculation.relax")
    calc_func = get_func_from_string(calc_func_str)

    return struct, worker, KPAR, NCORE, metadata, mpi_tasks, calc_func

def update_fworker(fw_doc, new_fworker):
    """
    Update the '_fworker' key in the Firework document to a new value.
    
    Args:
        fw_doc (dict): The Firework document to update.
        new_fworker (str): The new value for '_fworker'.
    
    Returns:
        dict: The updated Firework document.
    """
    # Find all paths to '_fworker'
    fw_doc["spec"]["_fworker"] = new_fworker
    fw_doc["spec"]["_tasks"][0]["job"]["fconfig"]["manager_config"]["_fworker"] = new_fworker
    fw_doc["spec"]["_tasks"][0]["job"]["config_updates"][0]["config"]["manager_config"]["_fworker"] = new_fworker
    return fw_doc

