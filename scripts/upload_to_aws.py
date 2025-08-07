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
    - Add a function (or incorporate a method into RunChgcarWF) to get data from HF dataset. 
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
        
        # Create S3 client with credentials
        session = botocore.session.get_session()
        s3_client = session.create_client(
            's3',
            region_name=self.region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            config=Config(signature_version='s3v4')
        )

        # check if a mat_id already exists in the S3 bucket
        try:
            mat_id = metadata['mat_id']
            fkey = '%s/LeMatRhoStaticMaker/CHGCAR.gz' %(mat_id)
            s3_client.head_object(Bucket=self.bucket_name, 
            Key=fkey)
            print('%s already exists in S3, skipping Flow' %(fkey))
            return None
        except botocore.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404' or error_code == 'NoSuchKey':
                print('%s does not exists in S3, proceeding with Flow' %(fkey))
                pass
            else:
                # For other errors, re-raise or handle as needed
                raise

        # Set up a two-step Flow object. boto_job will take the output of the relax_start_pbe job. 
        # The relax_start_pbe performs 4 DFT simulations: 
        # pre_static_maker, relax_maker_1, relax_maker_2, static_maker
        run_calc = relax_start_pbe(s, metadata)
        response = run_locally([run_calc], create_folders=True)

        # The boto_insert job will insert 4 sets of VASP calculations (one for each of the 4 
        # aforementioned DFT simulations). For pre_static_maker, relax_maker_1 and relax_maker_2 
        # we will only include the vasprun.xml an OUTCAR. For static_maker we will include 
        # everything but the WAVECAR and POTCAR.
        self.boto_insert(response, metadata)

    def boto_insert(
        self,
        response: Dict[str, Any], 
        metadata: Dict[str, Any],
        skip_files: Optional[list] = ["WAVECAR", "POTCAR", "POTCAR.orig"]) -> Any:
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
        
        # Create S3 client with credentials
        session = botocore.session.get_session()
        s3_client = session.create_client(
            's3',
            region_name=self.region_name,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            config=Config(signature_version='s3v4')
        )

        mat_id = metadata.get("mat_id", None)

        # sort the job responses by label_task 
        response_by_task_label = {}
        for uuid in response.keys():
            r = response[uuid][1]
            if r.output == None:
                continue
            response_by_task_label[r.output.task_label] = r
        
        # insert the first static calc
        file_path = response_by_task_label['LeMatRhoPreStaticMaker'].output.dir_name.split(':')[-1]
        for f in glob.glob(os.path.join(file_path, '*')):
            fname = f.split('/')[-1].replace('.gz', '')
            if "OUTCAR" not in f and "vasprun.xml" not in f:
                continue
            vjob_key = 'LeMatRhoPreStaticMaker'
            fkey = os.path.join(mat_id, vjob_key, f.split('/')[-2:][1])
            with open(f, 'rb') as body:
                s3_client.put_object(Bucket=self.bucket_name, Body=body, Key=fkey)
            print(f"File '{f}' uploaded to s3://{self.bucket_name}/{fkey}")

        # insert the first relax calc
        file_path = response_by_task_label['LeMatRhoRelaxMaker 1'].output.dir_name.split(':')[-1]
        for f in glob.glob(os.path.join(file_path, '*')):
            fname = f.split('/')[-1].replace('.gz', '')
            if "OUTCAR" not in f and "vasprun.xml" not in f:
                continue
            vjob_key = 'LeMatRhoRelaxMaker_1'
            fkey = os.path.join(mat_id, vjob_key, f.split('/')[-2:][1])
            with open(f, 'rb') as body:
                s3_client.put_object(Bucket=self.bucket_name, Body=body, Key=fkey)
            print(f"File '{f}' uploaded to s3://{self.bucket_name}/{fkey}")

        # insert the second relax calc
        file_path = response_by_task_label['LeMatRhoRelaxMaker 2'].output.dir_name.split(':')[-1]
        for f in glob.glob(os.path.join(file_path, '*')):
            fname = f.split('/')[-1].replace('.gz', '')
            if "OUTCAR" not in f and "vasprun.xml" not in f:
                continue
            vjob_key = 'LeMatRhoRelaxMaker_2'
            fkey = os.path.join(mat_id, vjob_key, f.split('/')[-2:][1])
            with open(f, 'rb') as body:
                s3_client.put_object(Bucket=self.bucket_name, Body=body, Key=fkey)
            print(f"File '{f}' uploaded to s3://{self.bucket_name}/{fkey}")

        # insert the second static calc
        file_path = response_by_task_label['LeMatRhoStaticMaker'].output.dir_name.split(':')[-1]
        for f in glob.glob(os.path.join(file_path, '*')):
            fname = f.split('/')[-1].replace('.gz', '')
            if fname in skip_files:
                continue
            vjob_key = 'LeMatRhoStaticMaker'
            fkey = os.path.join(mat_id, vjob_key, f.split('/')[-2:][1])
            with open(f, 'rb') as body:
                s3_client.put_object(Bucket=self.bucket_name, Body=body, Key=fkey)
            print(f"File '{f}' uploaded to s3://{self.bucket_name}/{fkey}")

        # insert metadata
        json.dump(metadata, open(os.path.join(file_path, 'metadata.json'), 'w'))
        with open(os.path.join(file_path, 'metadata.json'), 'rb') as body:
            s3_client.put_object(Bucket=self.bucket_name, Body=body, Key=fkey)
        fkey = os.path.join(mat_id, 'metadata.json')
        print(f"File '{f}' uploaded to s3://{self.bucket_name}/{fkey}")

        # insert jobflow outputs 
        output_dict = {}
        for uuid in response.keys():
            r = response[uuid][1]
            if r.output == None:
                continue
            doc = r.output.model_dump()
            doc = make_json_serializable(doc)
            output_dict[uuid] = doc
        json.dump(output_dict, open(os.path.join(file_path, 'response_outputs.json'), 'w'))        
        with open(os.path.join(file_path, 'response_outputs.json'), 'rb') as body:
            s3_client.put_object(Bucket=self.bucket_name, Body=body, Key=fkey)
        fkey = os.path.join(mat_id, 'response_outputs.json')
        print(f"File '{f}' uploaded to s3://{self.bucket_name}/{fkey}")


import datetime
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Composition
from emmet.core.symmetry import CrystalSystem

def make_json_serializable(obj):
    if isinstance(obj, dict):
        # Convert keys to str (or another string representation) and recursively convert values
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    elif isinstance(obj, datetime.date):
        return obj.isoformat()
    elif isinstance(obj, Element):
        return str(obj)
    elif isinstance(obj, Composition):
        return obj.as_dict()
    elif isinstance(obj, CrystalSystem):
        return str(obj)
    
    # Add additional custom conversions here, e.g., for sets, bytes, custom objects, etc.
    else:
        return str(obj)
