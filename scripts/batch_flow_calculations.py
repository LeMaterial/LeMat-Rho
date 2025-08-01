from pymatgen.core import Structure

from typing import Optional, Dict, Any

from run_calculation import relax_start_pbe
from upload_to_aws import boto_insert

from jobflow import Flow, SETTINGS, Response, job

def get_structure_from_hf_row(row):
    """Get a pymatgen Structure from a dictionary.
    The dictionary should contain the following keys:
        - lattice_vectors: list of lists containing the lattice vectors
        - species_at_sites: list of species at each site
        - cartesian_site_positions: list of cartesian site positions

    Parameters
    ----------
    row : dict
        Dictionary containing the structure information.

    Returns
    -------
    pymatgen.Structure
        Pymatgen Structure object.
    """

    return Structure(
        lattice=[x for y in row["lattice_vectors"] for x in y],
        species=row["species_at_sites"],
        coords=row["cartesian_site_positions"],
        coords_are_cartesian=True,
    )


@job
def run_multiple_calcs(
    batch_of_metadata: list,
    bucket_name: str,
    aws_access_key_id: Optional[str] = None, 
    aws_secret_access_key: Optional[str] = None, 
    region_name: Optional[str] = None,
    skip_files: Optional[list] = ["WAVECAR", "POTCAR"]) -> Any:
    """
    Inserts Completed VASP calculations into AWS S3 bucket.\
    
    batch_of_metadata::
        list of metadata for bulks to run calculations on
    bucket_name:: 
        name of the bucket
    object_key:: 
        string of text to be append to the front of the file, otherwise 
            the key will just be the full directory. e.g. 
            /path/to/VASP/calculation/CHGCAR is the Key if no object_key
            is given, otherwise it is /path/to/VASP/calculation/<object_key>_CHGCAR
    aws_access_key_id::
        aws access key
    aws_secret_access_key::
        aws secret access key
    region_name::
        name of region e.g. us-north-1    
    """

    all_flows = []
    for metadata in batch_of_metadata:
        s = get_structure_from_hf_row(
            lattice=[x for y in metadata["lattice_vectors"] for x in y],
            species=metadata["species_at_sites"],
            coords=metadata["cartesian_site_positions"],
            coords_are_cartesian=True,
        )
    
        run_calc = relax_start_pbe(s, metadata)
        boto_job = boto_insert(run_calc.output, metadata, bucket_name, 
                               aws_access_key_id, aws_secret_access_key, 
                               region_name, skip_files=skip_files)

        one_calc_flow = Flow([run_calc, boto_job], name=metadata['mat_id'], metadata=metadata)
        all_flows.append(one_calc_flow)

    return Flow(all_flows, name='batch_of_%s' %(len(batch_of_metadata)))