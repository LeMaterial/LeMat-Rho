import boto3
from botocore.exceptions import ClientError
import botocore.session
import botocore
from botocore.client import Config

from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.io import get_datafolder

from jobflow import Flow, SETTINGS, Response, job, run_locally, Job

from pymatgen.core import Structure

import os, argparse, json, sys, glob
from typing import Optional, Dict, Any
from pathlib import Path

from run_calculation import relax_start_pbe


"""
TODO:
    - Optimize core usage
    - Scrap Boto, use DataTrove for cleaner code
    - Job as json file, save some kind of record
    - Add a function (or incorporate a method into RunChgcarWF) to get data from HF dataset. 
    - Add a method to check S3 if data already exists
"""


class RunChgcarWF(PipelineStep):
    """
    DataTrove pipeline that manages Slurm submission of multiple JobFlow calculations in parallel. 
        Each job is associated with a bulk structure which will be relaxed using VASP and then 
        perform a static calculation to compute the CHGCAR and AECCAR files. The VASP ouput files
        are then stored in an AWS S3 bucket.

    Parameters
    ----------
    bucket_name : str
        AWS S3 Bucket name to insert data into
    aws_access_key_id : str
        AWS S3 access ID to insert data into
    aws_secret_access_key : str
        AWS S3 password to insert data into
    region_name : str
        Name of the region associated with your AWS S3 Bucket
    metadata_batch : list
        List of metadata with each item in the list being associated with one bulk (one set 
        of calculations).

    """


    def __init__(
        self, 
        bucket_name: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str,
        metadata_batch: list):
        super().__init__()

        """
        Attributes
        ----------
        bucket_name : str
            AWS S3 Bucket name to insert data into
        aws_access_key_id : str
            AWS S3 access ID to insert data into
        aws_secret_access_key : str
            AWS S3 password to insert data into
        region_name : str
            Name of the region associated with your AWS S3 Bucket
        metadata_batch : list
            List of metadata with each item in the list being associated with one bulk (one set 
            of calculations). The format of each metadata dictionary is as follows:


        """

        self.bucket_name = bucket_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self.metadata_batch = metadata_batch        

    def run(self, data, rank=0, world_size=1):
        """
        The run method is a part of the DataTrove syntax. By running this method with the 
            SlurmPipelineExecutor, it will count the rank starting from 0 to do an operation. 
            Here rank is utilized as the index of items in the metadata_batch. From the metadata, 
            we get the structure which is plugged into a Flow that calculates the CHGCAR 
            (relax_start_pbe) and inserts data into AWS S3 (boto_insert).
        """

        metadata = self.metadata_batch[rank]

        # make a PMG Structure object from metadata
        s = Structure(
        lattice=[x for y in metadata["lattice_vectors"] for x in y],
        species=metadata["species_at_sites"],
        coords=metadata["cartesian_site_positions"],
        coords_are_cartesian=True,
        )

        session = botocore.session.get_session()
        
        # Create S3 client with credentials
        s3_client = session.create_client(
            's3',
            region_name=self.region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            config=Config(signature_version='s3v4')
        )


        try:
            self.boto_check(metadata['mat_id'])
            print('%s already exists in S3, skipping Flow')
            return None
        except botocore.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404' or error_code == 'NoSuchKey':
                print('%s does not exists in S3, proceeding with Flow')
                pass
            else:
                # For other errors, re-raise or handle as needed
                raise


        # Set up a two-step Flow object. boto_job will take the output of the relax_start_pbe job. 
        # The relax_start_pbe performs 4 DFT simulations: 
        # pre_static_maker, relax_maker_1, relax_maker_2, static_maker
        run_calc = relax_start_pbe(s, metadata)

        # The boto_insert job will insert 4 sets of VASP calculations (one for each of the 4 
        # aforementioned DFT simulations). For pre_static_maker, relax_maker_1 and relax_maker_2 
        # we will only include the vasprun.xml an OUTCAR. For static_maker we will include 
        # everything but the WAVECAR and POTCAR.
        boto_job = boto_insert(
            s3_client,
            run_calc.output, self.bucket_name, 
            )

        run_locally([run_calc, boto_job], create_folders=True)

    def boto_check(self, mat_id):
        """
        Method to check if a mat_id already exists in the S3 bucket
        """

        fkey = '%s/static2/CHGCAR.gz' %(mat_id)
        self.s3_client.head_object(Bucket=self.bucket_name, 
        Key=fkey)


@job
def boto_insert(
                s3_session,
                prev_outputs: Dict[str, Any],
                bucket_name: str,
                skip_files: Optional[list] = ["WAVECAR", "POTCAR"]) -> Any:
    """
    Inserts Completed VASP calculations into AWS S3 bucket.

    file_path:: 
        directory of the VASP outputs
    bucket_name:: 
        name of the bucket
    object_key:: 
        string of text to be append to the front of the file, otherwise 
            the key will just be the full directory. e.g. 
            /path/to/VASP/calculation/CHGCAR is the Key if no object_key
            is given, otherwise it is /path/to/VASP/calculation/<object_key>_CHGCAR
    """


    metadata = prev_outputs['metadata']
    file_path = prev_outputs['relax_flow'].dir_name
    print('file_path: ', file_path)
    json.dump(metadata, open(os.path.join(file_path, 'metadata.json'), 'w'))

    mat_id = metadata.get("mat_id", None)

    vaspjobs = ['pre_static_job', 'relax_flow'] 

    for vaspjob in vaspjobs:
        file_path = prev_outputs[vaspjob].dir_name


        for f in glob.glob(os.path.join(file_path, '*')):
            fname = f.split('/')[-1].replace('.gz', '')
            if fname in skip_files:
                continue
            if vaspjob == 'pre_static_job':
                if "OUTCAR" not in f and "vasprun.xml" not in f:
                    continue
                vjob_key = 'static1'
            elif vaspjob == 'relax_flow':
                vjob_key = 'static2'

            fkey = os.path.join(mat_id, vjob_key, f.split('/')[-2:][1])
            with open(f, 'rb') as body:
                s3_session.put_object(Bucket=bucket_name, Body=body, Key=fkey)
            print(f"File '{f}' uploaded to s3://{bucket_name}/{fkey}")
