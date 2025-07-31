from atomate2.vasp.flows.mp import MPMetaGGADoubleRelaxStaticMaker
from atomate2.vasp.flows.matpes import MatPesStaticFlowMaker
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.mp import MPMetaGGARelaxMaker
from atomate2.vasp.jobs.matpes import MatPesMetaGGAStaticMaker, MatPesGGAStaticMaker
from atomate2.vasp.powerups import update_user_incar_settings, update_vasp_custodian_handlers

from pymatgen.core import Structure
from custodian.vasp.handlers import FrozenJobErrorHandler

from jobflow import SETTINGS
from jobflow import Flow
from jobflow.managers.local import run_locally

import boto3
from botocore.exceptions import ClientError

from typing import Optional, Dict, Any



"""Create and submit a workflow for relaxing a crystal structure using MatPES settings.
Important to set one consistent worker type for all calculations of a workflow due to copying of WAVECAR (requires same number of Cores with Ncore).
Four functions are provided:
1. relax: for double relaxation followed by a static calculation.
2  relax_start_pbe: for a static PBE calculation followed by an R2SCAN static calculation double relaxation followed by an R2SCAN static calculation.
3. static: for a single static calculation.
4. static_off_equilibrium: Usually off-equilibrium structures are harder to converge,
so this first does a PBE static calculation, then a meta-GGA static calculation.
update LargeSigmaHandler line 1427 with
            actions.append(
                {
                    "dict": "INCAR",
                    "action": {"_set": {"ICHARG": 1}},
                }
            )
"""


@job
def boto_insert(dir_name: str, bucket_name: str, 
    s3_prefix: Optional[str] = "", aws_access_key_id: Optional[str] = None, 
    aws_secret_access_key: Optional[str] = None) -> Any:
    """Inserts Completed VASP calculations into AWS S3 bucket."""

    # Use S3 client
    s3_client = boto3.client('s3')

    for root, _, files in os.walk(local_directory):
        for filename in files:
            # Full local path
            local_path = os.path.join(root, filename)
            # Relative path for S3 key
            relative_path = os.path.relpath(local_path, local_directory)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
            try:
                s3_client.upload_file(local_path, bucket_name, s3_key)
                print(f"Uploaded: {local_path} to s3://{bucket_name}/{s3_key}")
            except ClientError as e:
                print(f"Failed to upload {local_path}: {e}")


def relax(
    structure: Structure,
    metadata: Dict[str, Any],
    KPAR: int = 2,
    NCORE: int = 2,
    worker: Optional[str] = None,
    launchpad_path: Optional[str] = None
) -> None:
    """
    Create and submit a workflow for relaxing a crystal structure using a double relaxation
     followed by a static calculation all with MatPES settings.

    Parameters:
        structure (Structure):
            The pymatgen Structure object to relax.
        metadata (Dict[str, Any]):
            Additional metadata for the workflow. Must be provided and include a mat_id key.
            Ideally also previous ehull, energy, volume and band gap.
        KPAR (int, optional):
            K-point parallelization parameter. Default is 2.
        NCORE (int, optional):
            Number of cores per band calculation group. Default is 2.
        worker (str, optional):
            Worker identifier for the workflow manager.
        launchpad_path (str, optional):
            Path to the LaunchPad database. If not provided, it will use the default settings.
    Raises:
        ValueError: If mat_id is not provided in metadata.

    Returns:
        None: The workflow is submitted directly to the LaunchPad.
    """
    mat_id = metadata.get("mat_id", None)
    if mat_id is None:
        raise ValueError("Metadata must include a 'mat_id' key.")

    incar_settings = {"KPAR": KPAR, "NCORE": NCORE}

    incar_settings_relax = incar_settings.copy()
    incar_settings_relax.update({
        "ISIF": 3,
        "NSW": 100,
        "EDIFFG": -0.02,
        "IBRION": 2,
        "LWAVE": True,
    })
    # Setup relaxation makers
    relax_maker_1 = MatPesMetaGGAStaticMaker(
        name=f"LeMatRhoRelaxMaker",
        task_document_kwargs={"store_trajectory": True}
    )

    relax_maker_2 = MPMetaGGARelaxMaker(
        name=f"LeMatRhoRelaxMaker",
        copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")},
        task_document_kwargs={"store_trajectory": True}
    )

    double_relax_maker = DoubleRelaxMaker(
        name=f"LeMatRhoDoubleRelaxMaker",
        relax_maker1=relax_maker_1,
        relax_maker2=relax_maker_2
    )
    double_relax_maker = update_user_incar_settings(
        double_relax_maker,
        incar_settings_relax,
    )

    static_maker = MatPesMetaGGAStaticMaker(
        name=f"LeMatRhoStaticMaker",
        copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")},
    )
    static_maker = update_user_incar_settings(
        static_maker,
        incar_settings,
    )
    relax_workflow_maker = MPMetaGGADoubleRelaxStaticMaker(
        name=f"LeMatRhoDoubleRelaxStaticMaker",
        relax_maker=double_relax_maker,
        static_maker=static_maker
    )

    relax_flow = relax_workflow_maker.make(structure)

    # Set worker and metadata, most likely separate workers for different Nbands
    relax_flow.update_config({"manager_config": {"_fworker": worker}})
    relax_flow.update_metadata(metadata)

    # job to insert data into AWS S3
    boto_job = boto_insert()

     # Submit to JobFlow locally
    run_locally([relax_flow, boto_job])


def relax_start_pbe(
    structure: Structure,
    metadata: Dict[str, Any],
    KPAR: int = 2,
    NCORE: int = 2,
    worker: Optional[str] = None,
    launchpad_path: Optional[str] = None
) -> None:
    """
    Create and submit a workflow for relaxing a crystal structure using a 
     a static pbe calculation followed by a double relaxation
     followed by a static calculation all with MatPES settings.

    Parameters:
        structure (Structure):
            The pymatgen Structure object to relax.
        metadata (Dict[str, Any]):
            Additional metadata for the workflow. Must be provided and include a mat_id key.
            Ideally also previous ehull, energy, volume and band gap.
        KPAR (int, optional):
            K-point parallelization parameter. Default is 2.
        NCORE (int, optional):
            Number of cores per band calculation group. Default is 2.
        worker (str, optional):
            Worker identifier for the workflow manager.
        launchpad_path (str, optional):
            Path to the LaunchPad database. If not provided, it will use the default settings.
    Raises:
        ValueError: If mat_id is not provided in metadata.

    Returns:
        None: The workflow is submitted directly to the LaunchPad.
    """
    mat_id = metadata.get("mat_id", None)
    if mat_id is None:
        raise ValueError("Metadata must include a 'mat_id' key.")

    incar_settings = {"KPAR": KPAR, "NCORE": NCORE}

    incar_settings_relax = incar_settings.copy()
    incar_settings_relax.update({
        "ISIF": 3,
        "NSW": 100,
        "EDIFFG": -0.02,
        "IBRION": 2,
        "LWAVE": True,
    })
    # Setup relaxation makers
    incar_settings_pre_static = incar_settings.copy()
    incar_settings_pre_static.update({
        "LWAVE": True,
    })
    pre_static_maker = MatPesGGAStaticMaker(
        name=f"LeMatRhoPreStaticMaker"
    )
    pre_static_maker = update_user_incar_settings(
        pre_static_maker,
        incar_settings_pre_static,
    )

    relax_maker_1 = MatPesMetaGGAStaticMaker(
        name=f"LeMatRhoRelaxMaker",
        task_document_kwargs={"store_trajectory": True},
        copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")},
    )

    relax_maker_2 = MPMetaGGARelaxMaker(
        name=f"LeMatRhoRelaxMaker",
        copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")},
        task_document_kwargs={"store_trajectory": True},
    )

    double_relax_maker = DoubleRelaxMaker(
        name=f"LeMatRhoDoubleRelaxMaker",
        relax_maker1=relax_maker_1,
        relax_maker2=relax_maker_2
    )
    double_relax_maker = update_user_incar_settings(
        double_relax_maker,
        incar_settings_relax,
    )

    static_maker = MatPesMetaGGAStaticMaker(
        name=f"LeMatRhoStaticMaker",
        copy_vasp_kwargs={"additional_vasp_files": ("WAVECAR", "CHGCAR")},
    )
    static_maker = update_user_incar_settings(
        static_maker,
        incar_settings,
    )
    relax_workflow_maker = MPMetaGGADoubleRelaxStaticMaker(
        name=f"LeMatRhoDoubleRelaxStaticMaker",
        relax_maker=double_relax_maker,
        static_maker=static_maker,
    )
    pre_static_job = pre_static_maker.make(structure)
    relax_flow = relax_workflow_maker.make(
        pre_static_job.output.structure,
        prev_dir=pre_static_job.output.dir_name
    )
    complete_flow = Flow([pre_static_job, relax_flow])

    # Set worker and metadata, most likely separate workers for different Nbands
    complete_flow.update_config({"manager_config": {"_fworker": worker}})
    complete_flow.update_metadata(metadata)

    # job to insert data into AWS S3
    boto_job = boto_insert()

     # Submit to JobFlow locally
    run_locally([complete_flow, boto_job])


def static_calculation(
    structure: Structure,
    metadata: Dict[str, Any],
    KPAR: int = 2,
    NCORE: int = 2,
    worker: Optional[str] = None,
    launchpad_path: Optional[str] = None
) -> None:
    """
    Create and submit a workflow for a single static calculation using MatPES settings.

    Parameters:
        structure (Structure):
            The pymatgen Structure object to run a static calculation on.
        metadata (Dict[str, Any]):
            Additional metadata for the workflow. Must include a mat_id key.
        KPAR (int, optional):
            K-point parallelization parameter. Default is 2.
        NCORE (int, optional):
            Number of cores per band calculation group. Default is 2.
        worker (str, optional):
            Worker identifier for the workflow manager.
        launchpad_path (str, optional):
            Path to the LaunchPad database. If not provided, it will use the default settings.

    Raises:
        ValueError: If metadata is not provided.

    Returns:
        None: The workflow is submitted directly to the LaunchPad.
    """
    mat_id = metadata.get("mat_id")
    if not mat_id:
        raise ValueError("Metadata must include a 'mat_id' key.")

    # INCAR settings
    incar_settings = {"KPAR": KPAR, "NCORE": NCORE}

    # Set up static calculation maker
    static_maker = MatPesMetaGGAStaticMaker(
        name=f"LeMatRhoStaticMaker",
    )

    static_maker = update_user_incar_settings(
        static_maker,
        incar_settings,
    )

    static_flow = static_maker.make(structure)

    # Apply config and metadata
    static_flow.update_config({"manager_config": {"_fworker": worker}})
    static_flow.update_metadata(metadata)

    # job to insert data into AWS S3
    boto_job = boto_insert()

     # Submit to JobFlow locally
    run_locally([static_flow, boto_job])



def static_calculation_off_equilibrium(
    structure: Structure,
    metadata: Dict[str, Any],
    KPAR: int = 2,
    NCORE: int = 2,
    worker: Optional[str] = None,
    launchpad_path: Optional[str] = None
) -> None:
    """
    Create and submit a workflow for a single static calculation using MatPES settings.

    Parameters:
        structure (Structure):
            The pymatgen Structure object to run a static calculation on.
        metadata (Dict[str, Any]):
            Additional metadata for the workflow. Must include a mat_id key.
        KPAR (int, optional):
            K-point parallelization parameter. Default is 2.
        NCORE (int, optional):
            Number of cores per band calculation group. Default is 2.
        worker (str, optional):
            Worker identifier for the workflow manager.
        launchpad_path (str, optional):
            Path to the LaunchPad database. If not provided, it will use the default settings.
    Raises:
        ValueError: If metadata is not provided.

    Returns:
        None: The workflow is submitted directly to the LaunchPad.
    """
    mat_id = metadata.get("mat_id")
    if not mat_id:
        raise ValueError("Metadata must include a 'mat_id' key.")

    # INCAR settings
    incar_settings = {"KPAR": KPAR, "NCORE": NCORE}

    # Set up static calculation maker
    static_maker = MatPesStaticFlowMaker(
        name=f"LeMatRhoStaticMaker",
        static3=None
    )

    static_maker = update_user_incar_settings(
        static_maker,
        incar_settings,
    )

    static_flow = static_maker.make(structure)

    # Apply config and metadata
    static_flow.update_config({"manager_config": {"_fworker": worker}})
    static_flow.update_metadata(metadata)

    # job to insert data into AWS S3
    boto_job = boto_insert()

     # Submit to JobFlow locally
    run_locally([static_flow, boto_job])