import boto3
import gzip
import io
from botocore import UNSIGNED
from botocore.config import Config
import json

from pymatgen.io.vasp import Chgcar

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

    def to_hf_data(self):
        pass


if __name__ == "__main__":
    stream_gz_file_from_aws_bucket(
        s3_key="chgcars/mp-1000002.json.gz", processor_cls=ChgCarProcessor
    )
