from upload_to_aws import boto_insert
from run_calculation import relax_start_pbe
from jobflow import run_locally
from pymatgen.core.structure import Structure

import os, argparse


def read_options():

    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--bucket_name", dest="bucket_name", type=str, 
                        help="name of aws s3 bucket")
    parser.add_argument("-a", "--aws_access_key_id", dest="aws_access_key_id", type=str, 
                        help="aws s3 access ID")
    parser.add_argument("-s", "--aws_secret_access_key", dest="aws_secret_access_key", type=str, 
                        help="aws s3 secret access key")
    parser.add_argument("-r", "--region_name", dest="region_name", type=str, 
                        help="aws s3 region name")
    parser.add_argument("-b", "--batch_file", dest="batch_file", type=str, 
                        help="File containing dicts of metadata")
    parser.add_argument("-i", "--batch_index", dest="batch_index", type=int, 
                        help="Index of item in the json file corresponding what structure to run calculation on")
    

    args = parser.parse_args()

    return args


if __name__=="__main__":

    args = read_options() 

    bucket_name = args.bucket_name
    aws_access_key_id = args.aws_access_key_id
    aws_secret_access_key = args.aws_secret_access_key
    region_name = args.region_name
    batch_file = args.batch_file
    batch_index = args.batch_index

    if batch_index == 'test':
        s = Structure.from_file('Li.cif')
        metadata = {'mat_id': 'Li_test'}
    else:
        batch_metadata = json.load(open(batch_file, 'r'))
        metadata = batch_metadata[batch_index]
        s = get_structure_from_hf_row(
            lattice=[x for y in metadata["lattice_vectors"] for x in y],
            species=metadata["species_at_sites"],
            coords=metadata["cartesian_site_positions"],
            coords_are_cartesian=True,
        )


    run_calc = relax_start_pbe(s, metadata)
    boto_job = boto_insert(run_calc.output, bucket_name, 
                           aws_access_key_id, aws_secret_access_key, region_name)
    run_locally([run_calc, boto_job], create_folders=True)
