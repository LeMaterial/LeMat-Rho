from pymatgen.core import Structure

from typing import Optional, Dict, Any

from run_calculation import relax_start_pbe
from upload_to_aws import boto_insert

from jobflow import Flow, SETTINGS, Response, job, run_locally

from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline

import os, argparse, json


class RunChgcarWF(PipelineStep):

    def __init__(self, bucket_name, aws_access_key_id, aws_secret_access_key,
    region_name, metadata):
        super().__init__()

        self.bucket_name = bucket_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self.metadata = metadata

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:

        s = Structure(
        lattice=[x for y in metadata["lattice_vectors"] for x in y],
        species=metadata["species_at_sites"],
        coords=metadata["cartesian_site_positions"],
        coords_are_cartesian=True,
        )

        run_calc = relax_start_pbe(s, metadata)
        boto_job = boto_insert(run_calc.output, metadata, self.bucket_name, 
                                self.aws_access_key_id, self.aws_secret_access_key, 
                                self.region_name)

        run_locally([run_calc, boto_job], create_folders=True)
