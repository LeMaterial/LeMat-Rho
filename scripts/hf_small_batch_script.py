from upload_to_aws import boto_insert
from run_calculation import relax_start_pbe
from jobflow import run_locally
from pymatgen.core.structure import Structure

import os, argparse, json

from datatrove.pipeline.base import PipelineStep
from datatrove.executor import SlurmPipelineExecutor

from batch_flow_calculations import RunChgcarWF


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
    parser.add_argument("-f", "--batch_file", dest="batch_file", type=str, 
                        help="File containing dicts of metadata")
    parser.add_argument("-l", "--logdir", dest="logdir", type=str, 
                        help="Directory to log outputs")
    parser.add_argument("-p", "--partition", dest="partition", type=str, 
                        help="partition")
    parser.add_argument("-c", "--cpus_per_task", dest="cpus_per_task", type=str, 
                        help="cpus_per_task")
    parser.add_argument("-e", "--env_command", dest="env_command", type=str, 
                        help="Location of file to activate environment")
    
    args = parser.parse_args()

    return args


if __name__=="__main__":

    args = read_options() 

    bucket_name = args.bucket_name
    aws_access_key_id = args.aws_access_key_id
    aws_secret_access_key = args.aws_secret_access_key
    region_name = args.region_name
    batch_file = args.batch_file
    logdir = args.logdir
    partition = args.partition
    cpus_per_task = args.cpus_per_task
    env_command = args.env_command

    metadata_batch = json.load(open(batch_file, 'r'))

    for i, metadata in enumerate(metadata_batch):

        SlurmPipelineExecutor(
            pipeline=[
                RunChgcarWF(
                    bucket_name, 
                    aws_access_key_id, 
                    aws_secret_access_key,
                    region_name, 
                    metadata
                )
            ],
            job_name="small_batch_RunChgcarWF",
            logging_dir=logdir,
            partition=partition,
            # sbatch_args={
            #     "mem-per-cpu": "1950M"
            # },
            env_command='source %s' %(env_command),
            cpus_per_task=cpus_per_task,
            tasks=1,
            max_array_launch_parallel=True,
            time="03:00:00",
        ).run()
