import boto3
from botocore.exceptions import ClientError
import botocore.session
from botocore.client import Config
import glob, os
from pathlib import Path

from jobflow import job
from typing import Optional, Dict, Any


@job
def boto_insert(file_path: str, 
                metadata: Dict[str, Any],
                bucket_name: str,
                aws_access_key_id: Optional[str] = None, 
                aws_secret_access_key: Optional[str] = None, 
                region_name: Optional[str] = None,
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
    aws_access_key_id::
        aws access key
    aws_secret_access_key::
        aws secret access key
    region_name::
        name of region e.g. us-north-1    
    """

    session = botocore.session.get_session()
    
    # Create S3 client with credentials
    s3 = session.create_client(
        's3',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        config=Config(signature_version='s3v4')
    )

    mat_id = metadata.get("mat_id", None)
    for f in glob.glob(os.path.join(file_path, '*')):
        fname = f.split('/')[-1].replace('.gz', '')
        if fname in skip_files:
            continue
        fkey = os.path.join(mat_id, f.split('/')[-2:][0], f.split('/')[-2:][1])
        with open(f, 'rb') as body:
            s3.put_object(Bucket=bucket_name, Body=body, Key=fkey)
        print(f"File '{f}' uploaded to s3://{bucket_name}/{fkey}")

