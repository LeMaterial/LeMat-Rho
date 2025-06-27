'''
HF DataFrame Rows

| ID | LeMat-Bulk ID | BAWL Hash | Functional | Lattice Vectors | Species at Sites | Cartesian Site Positions | Normalized charge density | Normalized AECCAR0 | Normalized AECCAR1 | Normalized AECCAR2 | Bader Charge Partition | DDEC6 Charge Partition |

'''

import boto3
import gzip
import io
from botocore import UNSIGNED
from botocore.config import Config
import json

from pymatgen.io.vasp import Chgcar

from pyrho.pgrid import PGrid
from pyrho.charge_density import ChargeDensity

AWS_BUCKET_NAME = "materialsproject-parsed"


def stream_gz_file_from_aws_bucket(s3_key, processor_cls):
    s3 = boto3.client(
        "s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED)
    )
    response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=s3_key)

    # Stream and decompress using GzipFile
    gzipped_body = gzip.GzipFile(fileobj=response["Body"])

    # If needed, wrap in BufferedReader to make it seekable
    buffered_reader = io.BufferedReader(gzipped_body)

    # Pass to processor
    processor = processor_cls(buffered_reader)
    processor.process()


class ChgCarProcessor:
    def __init__(self, file_obj):
        self.file_obj = file_obj

    def process(self):
        self.chgcar = Chgcar.from_dict(
            json.loads(self.file_obj.read().decode("utf-8"))["data"]
        )
        self.to_hf_data()

    def to_hf_data(self):
        chg_density = ChargeDensity.from_pmg(self.chgcar)
        print(self.chgcar.structure)
        for pgrid in chg_density.pgrids.values():
            print(pgrid.lossy_smooth_compression([30,30,30]))


if __name__ == "__main__":
    print(stream_gz_file_from_aws_bucket(
        s3_key="chgcars/mp-1000002.json.gz", processor_cls=ChgCarProcessor
    ))
