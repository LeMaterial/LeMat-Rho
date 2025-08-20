"""
HF DataFrame Rows

| ID | LeMat-Bulk ID | BAWL Hash | Functional | Lattice Vectors |
| Species at Sites | Cartesian Site Positions | Normalized charge density |
| Normalized AECCAR0 | Normalized AECCAR1 | Normalized AECCAR2 |
| Bader Charge Partition | DDEC6 Charge Partition |
"""

import boto3
import gzip
import io
import os
import pandas as pd

from pymatgen.io.vasp import Chgcar
from pymatgen.core import Structure
from pymatgen.command_line.bader_caller import BaderAnalysis

from monty.tempfile import ScratchDir
import numpy as np
from datasets import Dataset

from pyrho.charge_density import ChargeDensity
from material_hasher.hasher.bawl import BAWLHasher

from pymatgen.io.vasp import Potcar, Vasprun
import subprocess
import sys
from pymatgen.io.vasp.sets import MatPESStaticSet


AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_BUCKET_NAME = "lemat-rho"

PERL_CHGCARSUM_FILE = "/Users/martinsiron/Downloads/vtstscripts-1034/chgsum.pl"
BADER_PATH = "/Users/martinsiron/Downloads/bader_osx"  # bader executable


def pymatgen_to_optimade(pmg_structure: Structure):
    data = {}
    data["elements"] = pmg_structure.chemical_system_set
    data["nsites"] = len(pmg_structure)
    data["chemical_formula_anonymous"] = pmg_structure.composition.anonymized_formula
    data["chemical_formula_reduced"] = (
        pmg_structure.composition.reduced_composition.to_pretty_string()
    )
    data["chemical_formula_descriptive"] = pmg_structure.composition.to_pretty_string()
    data["nelements"] = len(pmg_structure.chemical_system_set)
    data["dimension_types"] = [1, 1, 1]
    data["nperiodic_dimensions"] = 3
    data["lattice_vectors"] = pmg_structure.lattice.matrix
    data["cartesian_site_positions"] = pmg_structure.cart_coords
    data["species_at_sites"] = [x.name for x in pmg_structure.elements]
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


# calc_type = 'LeMatRhoRelaxMaker_1'
def stream_gz_folder_from_aws_bucket(
    folder, files, calculation_type="LeMatRhoStaticMaker"
):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    for file in files:
        response = s3.get_object(
            Bucket=AWS_BUCKET_NAME, Key=os.path.join(folder, calculation_type, file)
        )
        gzipped_body = gzip.GzipFile(fileobj=response["Body"])
        buffered_reader = io.BufferedReader(gzipped_body)
        with open(file.split(".")[0], "w") as fh:
            fh.write(buffered_reader.read().decode("utf-8"))


class CubeProcessor:
    def __init__(
        self,
        file_name,
        cube_class,
        key,
        pgrid_key="total",
        compression_shape=[200, 200, 200],
    ):
        self.cube_class = cube_class
        self.file_name = file_name
        self.pgrid_key = pgrid_key
        self.compression_shape = compression_shape
        self._hf_data = {}
        # Files in our bucket are gzipped CHGCAR-like text files

        self.cube_obj = cube_class.from_file(file_name)
        self.key = key

    def process(self):
        density = ChargeDensity.from_pmg(self.cube_obj)
        pgrid = density.pgrids[self.pgrid_key]
        self._hf_data.update(
            {
                f"{self.key}": density.pgrids[self.pgrid_key].grid_data,
                "compressed_charge_density": (
                    pgrid.lossy_smooth_compression(self.compression_shape)
                ),
            }
        )

    @property
    def grid_3d(self):
        density = ChargeDensity.from_pmg(self.cube_obj)
        return density.pgrids[self.pgrid_key].grid_data


class ChgCarProcessor(CubeProcessor):
    def __init__(
        self,
        file_name,
        key="charge_density",
        cube_class=Chgcar,
        pgrid_key="total",
        compression_shape=[15, 15, 15],
    ):
        super().__init__(
            file_name,
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
        with ScratchDir('.', ) as sd:
            try:
                print(f"processing {material_id}")
                row = {}
                row.update({"immutable_id": material_id})
                # Download files

                stream_gz_folder_from_aws_bucket(
                    material_id, ["vasprun.xml.gz"], calculation_type="LeMatRhoRelaxMaker_1"
                )
                stream_gz_folder_from_aws_bucket(
                    material_id,
                    ["CHGCAR.gz", "AECCAR0.gz", "AECCAR1.gz", "AECCAR2.gz"],
                )

                # Process CHGCAR:
                chgcar = ChgCarProcessor("CHGCAR", cube_class=Chgcar)
                chgcar.process()
                row["bawl_hasher"] = bh.get_material_hash(chgcar.cube_obj.structure)
                row.update(pymatgen_to_optimade(chgcar.cube_obj.structure))
                row["compressed_charge_density"] = chgcar.grid_3d

                aeccar0 = ChgCarProcessor("AECCAR0", cube_class=Chgcar)
                aeccar0.process()
                row["compressed_aeccar0_density"] = chgcar.grid_3d

                aeccar1 = ChgCarProcessor("AECCAR1", cube_class=Chgcar)
                aeccar1.process()
                row["compressed_aeccar1_density"] = aeccar1.grid_3d

                aeccar2 = ChgCarProcessor("AECCAR2", cube_class=Chgcar)
                aeccar2.process()
                row["compressed_aeccar2_density"] = aeccar2.grid_3d

                ## Bader Charge Partitioning
                MatPESStaticSet(Vasprun("vasprun").structures[0]).potcar.write_file("POTCAR")
                params = ["AECCAR0", "AECCAR2"]
                perl_script = subprocess.Popen(
                    [PERL_CHGCARSUM_FILE, *params], stdout=sys.stdout
                )
                perl_script.communicate()
                ba = BaderAnalysis(
                    os.path.join(os.getcwd(), "CHGCAR"),
                    os.path.join(os.getcwd(), "POTCAR"),
                    chgref_filename=os.path.join(os.getcwd(), "CHGCAR_sum"),
                    bader_path=BADER_PATH,
                )
                row["bader_charges"] = ba.get_charge_decorated_structure().site_properties[
                    "charge"
                ]
                data.append(row)
            except:
                continue

    df = pd.DataFrame(data)
    push_dataframe_to_hf_dataset(df, "lematerial/LeMat-Rho", private=True)
