from upload_to_aws import boto_insert
from run_calculation import relax_start_pbe
from jobflow import run_locally
from pymatgen.core.structure import Structure

import os, argparse, json


def read_options():

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file_path", dest="file_path", type=str, 
                        help="path to the vasp folder")
    parser.add_argument("-b", "--bucket_name", dest="bucket_name", type=str, 
                        help="name of aws s3 bucket")
    parser.add_argument("-i", "--aws_access_key_id", dest="aws_access_key_id", type=str, 
                        help="aws s3 access ID")
    parser.add_argument("-s", "--aws_secret_access_key", dest="aws_secret_access_key", type=str, 
                        help="aws s3 secret access key")
    parser.add_argument("-r", "--region_name", dest="region_name", type=str, 
                        help="aws s3 region name")


    args = parser.parse_args()

    return args


if __name__=="__main__":

    args = read_options() 

    file_path = args.file_path
    bucket_name = args.bucket_name
    aws_access_key_id = args.aws_access_key_id
    aws_secret_access_key = args.aws_secret_access_key
    region_name = args.region_name


    small_batch = json.load(open('small_batch_800.json', 'r'))

    run_calc = relax_start_pbe(s, metadata)
    boto_job = boto_insert(run_calc.dir_name, metadata, file_path, bucket_name, 
                           aws_access_key_id, aws_secret_access_key, region_name)
    run_locally([run_calc, boto_job], create_folders=True)
