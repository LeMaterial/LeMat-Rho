from datatrove.executor import SlurmPipelineExecutor

from upload_to_aws import RunChgcarWF

import argparse, json


def read_options():

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--cpus_per_task", dest="cpus_per_task", type=str, 
                        help="cpus_per_task", default=2)
    parser.add_argument("-p", "--partition", dest="partition", type=str, 
                        help="partition", default="hopper-cpu")
    parser.add_argument("-t", "--time", dest="time", type=str, 
                        help="time")
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
    parser.add_argument("-e", "--env_command", dest="env_command", type=str, 
                        help="Location of file to activate environment")
    
    args = parser.parse_args()

    return args


if __name__=="__main__":

    args = read_options() 

    # batch script inputs
    cpus_per_task = args.cpus_per_task
    partition = args.partition
    t = args.time
    all_env_commands = ["conda activate rho", # @Georgia, please check if this is the correct way to soruce the envs
                        'source /fsx/georgia_channing/LeMat-Rho/.venv/bin/activate' %(args.env_command), # @Georgia, please check path
                        'export PATH=/fsx/georgia_channing/VASP/exe/IntelMPI2019/Linux-x86_64/bin:$PATH',
                        'export PATH=/fsx/georgia_channing/VASP/exe/vasp6.4.3_cml/Linux-x86_64:$PATH',
                        'export LD_LIBRARY_PATH=/fsx/georgia_channing/VASP/exe/IntelMPI2019/Linux-x86_64/libfabric/lib/prov:/fsx/georgia_channing/VASP/exe/IntelMPI2019/Linux-x86_64/libfabric/lib:/fsx/georgia_channing/VASP/exe/IntelMPI2019/Linux-x86_64/lib:$LD_LIBRARY_PATH',
                        'which vasp_std',
                        'which mpirun',
                        'export VASP_CMD="mpirun /fsx/georgia_channing/VASP/exe/vasp6.4.3_cml/Linux-x86_64/vasp_std"'
                       ]
    env_command =' && '.join(all_env_commands)
    
    logdir = args.logdir

    # inputs for jobflow workflow
    bucket_name = args.bucket_name
    aws_access_key_id = args.aws_access_key_id
    aws_secret_access_key = args.aws_secret_access_key
    region_name = args.region_name
    batch_file = args.batch_file

    metadata_batch = json.load(open(batch_file, 'r'))

    SlurmPipelineExecutor(
        pipeline=[
            RunChgcarWF(
                bucket_name, 
                aws_access_key_id, 
                aws_secret_access_key,
                region_name, 
                metadata_batch,
            )
        ],
        job_name="small_batch_RunChgcarWF",
        logging_dir=logdir,
        partition=partition,
        sbatch_args={"qos": 'normal',
            "mem-per-cpu": "2G"
        },
        env_command=env_command,
        cpus_per_task=cpus_per_task,
        tasks=len(metadata_batch),
        max_array_launch_parallel=True,
        time=t,
    ).run()