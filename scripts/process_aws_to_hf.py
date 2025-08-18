''' 
HF DataFrame Rows

| ID | LeMat-Bulk ID | BAWL Hash | Functional | Lattice Vectors |
| Species at Sites | Cartesian Site Positions | Normalized charge density |
| Normalized AECCAR0 | Normalized AECCAR1 | Normalized AECCAR2 |
| Bader Charge Partition | DDEC6 Charge Partition |
'''

import boto3
import gzip
import io
import os
import pandas as pd

from pymatgen.io.vasp import Chgcar
from pymatgen.core import Structure

import tempfile
import numpy as np
from datasets import Dataset

from pyrho.charge_density import ChargeDensity
from material_hasher.hasher.bawl import BAWLHasher

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = "lemat-rho"


def pymatgen_to_optimade(pmg_structure: Structure):
    data = {}
    data['elements'] = pmg_structure.chemical_system_set
    data['nsites'] = len(pmg_structure)
    data['chemical_formula_anonymous'] = (
        pmg_structure.composition.anonymized_formula
    )
    data['chemical_formula_reduced'] = (
        pmg_structure.composition.reduced_composition.to_pretty_string()
    )
    data['chemical_formula_descriptive'] = (
        pmg_structure.composition.to_pretty_string()
    )
    data['nelements'] = len(pmg_structure.chemical_system_set)
    data['dimension_types'] = [1, 1, 1]
    data['nperiodic_dimensions'] = 3
    data['lattice_vectors'] = pmg_structure.lattice.matrix
    data['cartesian_site_positions'] = pmg_structure.cart_coords
    data['species_at_sites'] = [x.name for x in pmg_structure.elements]
    return data


def push_dataframe_to_hf_dataset(
    df,
    repo_id: str,
    split: str = "train",
    private: bool = False,
    token: str | None = None,
):
    """Push a pandas DataFrame to a Hugging Face dataset repository.

    - Converts numpy arrays in cells (e.g., 3D grids) to nested Python lists.
    - Converts numpy scalar types to native Python types.
    - Creates a `datasets.Dataset` and pushes it to the Hub.
    """

    def to_serializable(value):
        if isinstance(value, np.ndarray):
            # Use float32 to reduce size; convert to nested lists
            if value.dtype != np.float32:
                value = value.astype(np.float32)
            return value.tolist()
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        return value

    df_serializable = df.applymap(to_serializable)
    dataset = Dataset.from_pandas(df_serializable, preserve_index=False)
    dataset.push_to_hub(
        repo_id=repo_id,
        split=split,
        private=private,
        token=token or os.getenv("HF_TOKEN"),
    )
    return dataset


def stream_gz_file_from_aws_bucket(s3_key, processor_cls, **processor_kwargs):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=s3_key)

    # Stream and decompress using GzipFile
    gzipped_body = gzip.GzipFile(fileobj=response["Body"])

    # If needed, wrap in BufferedReader to make it seekable
    buffered_reader = io.BufferedReader(gzipped_body)

    # Pass to processor
    processor = processor_cls(buffered_reader, **processor_kwargs)

    # Do not call processor.process() here, just return the processor
    return processor


class CubeProcessor:
    def __init__(
        self,
        file_obj,
        cube_class,
        key,
        pgrid_key='total',
        compression_shape=[200, 200, 200],
    ):
        self.cube_class = cube_class
        self.file_obj = file_obj
        self.pgrid_key = pgrid_key
        self.compression_shape = compression_shape
        self._hf_data = {}
        # Files in our bucket are gzipped CHGCAR-like text files
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmp:
            tmp.write(file_obj.read().decode('utf-8'))

            self.cube_obj = cube_class.from_file(
                tmp.name
            )
        self.key = key

    def process(self):
        density = ChargeDensity.from_pmg(self.cube_obj)
        pgrid = density.pgrids[self.pgrid_key]
        self._hf_data.update({
            f'{self.key}': density.pgrids[self.pgrid_key].grid_data,
            'compressed_charge_density': (
                pgrid.lossy_smooth_compression(self.compression_shape)
            ),
        })

    @property
    def grid_3d(self):
        density = ChargeDensity.from_pmg(self.cube_obj)
        return density.pgrids[self.pgrid_key].grid_data


class AeccarProcessor(CubeProcessor):
    def __init__(
        self,
        file_obj,
        key='aeccar0',
        cube_class=Chgcar,
        pgrid_key='total',
        compression_shape=[20, 20, 20],
    ):
        super().__init__(
            file_obj,
            cube_class=cube_class,
            key=key,
            pgrid_key=pgrid_key,
            compression_shape=compression_shape,
        )


class ChgCarProcessor(CubeProcessor):
    def __init__(
        self,
        file_obj,
        key='charge_density',
        cube_class=Chgcar,
        pgrid_key='total',
        compression_shape=[15, 15, 15],
    ):
        super().__init__(
            file_obj,
            cube_class=cube_class,
            key=key,
            pgrid_key=pgrid_key,
            compression_shape=compression_shape,
        )


if __name__ == "__main__":
    bh = BAWLHasher()
    # get list of all folders in bucket using secret key
    material_ids = []
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    response = s3.list_objects_v2(Bucket=AWS_BUCKET_NAME, Delimiter="/")
    if "CommonPrefixes" in response:
        for prefix in response["CommonPrefixes"]:
            if (
                "oqmd-" in prefix["Prefix"]
                or "mp-" in prefix["Prefix"]
                or "agm" in prefix["Prefix"]
            ):
                material_ids.append(prefix["Prefix"][:-1])

    data = []
    for material_id in material_ids:
        try:
            print(f'processing {material_id}')
            row = {}
            row.update({'immutable_id': material_id})
            # Normalized charge density
            chgcar = stream_gz_file_from_aws_bucket(
                s3_key=f"{material_id}/LeMatRhoStaticMaker/CHGCAR.gz",
                processor_cls=ChgCarProcessor,
                key='charge_density',
            )
            chgcar.process()
            row['bawl_hasher'] = bh.get_material_hash(chgcar.cube_obj.structure)
            row.update(pymatgen_to_optimade(chgcar.cube_obj.structure))
            row['compressed_charge_density'] = chgcar.grid_3d
            aeccar0 = stream_gz_file_from_aws_bucket(
                s3_key=f"{material_id}/LeMatRhoStaticMaker/AECCAR0.gz",
                processor_cls=ChgCarProcessor,
                key='aeccar0',
            )
            aeccar0.process()
            row['compressed_aeccar0_density'] = chgcar.grid_3d
            aeccar0 = stream_gz_file_from_aws_bucket(
                s3_key=f"{material_id}/LeMatRhoStaticMaker/AECCAR1.gz",
                processor_cls=ChgCarProcessor,
                key='aeccar1',
            )
            aeccar0.process()
            row['compressed_aeccar1_density'] = chgcar.grid_3d
            aeccar0 = stream_gz_file_from_aws_bucket(
                s3_key=f"{material_id}/LeMatRhoStaticMaker/AECCAR2.gz",
                processor_cls=ChgCarProcessor,
                key='aeccar2',
            )
            aeccar0.process()
            row['compressed_aeccar2_density'] = chgcar.grid_3d
            data.append(row)
        except:
            print(f"failed on {material_id}")
            continue

    df = pd.DataFrame(data)
    push_dataframe_to_hf_dataset(df, 'lematerial/LeMat-Rho', private=True)
