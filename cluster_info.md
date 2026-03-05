Slurm Quick Start Guide
This guide provides some quick information to help you get started using Slurm in SoC Compute Cluster. Please refer to the official Slurm documentation for more comprehensive information. It is not our intention to duplicate the official documentation here. You can also find some tutorials from the Slurm website.

You have to login to the Slurm login nodes (xlogin0, xlogin1, or xlogin2) to run these commands.

For users familiar with the legacy compute cluster environment, the Moving from Legacy Compute Cluster to Slurm guide may help with your transition.

General Cluster Commands
sinfo: Show status of nodes and partitions in the Slurm cluster.

squeue: Show the Slurm job queue.

Submitting Batch Jobs
Batch submission is the primary and prefered method of submitting jobs to Slurm. In batch submission, you have to write a batch script to run your job. This can include multiple commands in multiple steps, and you have full flexibility in determining how the job will be executed. The script is submitted to Slurm for queuing, and you can logoff immediately. Slurm will schedule the job automatically.

Note: Slurm batch jobs are simply Unix shell scripts. There is no new scripting language to learn. You must have at least some basic familiarity with writing shell scripts to get something useful done with Slurm.

sbatch : Submit batch script to Slurm.

A simple batch script example:

#!/bin/sh

srun sleep 60
Name the script sleep.sh. Then submit the job with this command: sbatch sleep.sh. If you want to run 10 instances of this sleep 60, then submit the job as: sbatch -n 10 sleep.sh.

When your batch job completes, Slurm writes its terminal output to the slurm-<jobid>.out file.

There are many arguments that you can give to sbatch to specify different job requirements, such as requirement for processors, distinct nodes, need for GPU, etc. Some of these arguments can also be specified inside the batch script like in the following:

#!/bin/sh
#SBATCH --time=12
#SBATCH --job-name=mytestjob
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user@comp.nus.edu.sg

srun someprogram
The above specified a time limit of 12 minutes, a job name, and that email be sent on the specified job events.

You can ask for a GPU allocation by adding the -G 1 argument to sbatch. For example:

#!/bin/sh
#SBATCH --job-name=gpujob
#SBATCH --gpus=1

srun somegpuprogram
The above asks for one GPU from a node in the standard partition. You may also request allocation of specific GPU types with something like –gpus=a100:1. Currently, specific types like a100, t4, and titan may be specified. The list of GPU types can be found at the Slurm Info page; you can also refer to our hardware inventory.

You can also develop your script to perform some job preparation functions before your main compute job runs, or perform some cleanup or results collection after the job runs. Here's an example:

#!/bin/sh

# Do some job preparation. E.g. copy data to /temp.
TMPDIR=`mktemp -d`
cp ~/jobdata/* $TMPDIR

# Do the job and tell it where to find the working directory.
srun someprogram $TMPDIR

# Copy out the results and cleanup.
cp $TMPDIR/* ~/jobresults/
rm -rf $TMPDIR
You might want to do this if you want your compute job to read/write data to the respective compute node's local storage. Please remember to copy out files from the local storage you need, because otherwise you may not easily get back access to the specific compute node to get the files. Leftover files in local storage will be removed from time to time.

Not surprisingly, you shoud be familiar with shell scripting, or even reasonably proficient with it, in order to submit non-trivial batch jobs to Slurm. There are many resources on the Internet to learn about shell scripting. We have a few suggestions listed to learn more about Unix.

Run Adhoc Jobs
You can also run jobs on an adhoc basis. Note that such jobs require you to release the allocation when you are done, otherwise they will run until timeout. Please note that any idle time is accrued against your fairshare usage and thus deprioritises your subsequent job allocations.

These commands require you to wait for an allocation to suceed before you can continue to issue any commands. If the resources you require are not available, you'll have to wait. This is unlike the sbatch command which returns immediately and automatically executes your job in accordance to the Slurm scheduler algorithm.

salloc: Obtain a job allocation, execute commands, then release the allocation. This allows you to pseudo-interactively run a job once an allocation is available.

srun: Within a job allocation, dispatch commands in parallel. When run outside a job allocation, this command can automatically obtain an allocation.

Here's an example:

user@xlogin0:~$ salloc
salloc: Granted job allocation 7731
user@xlogin0:~$ srun hostname
xcnb0
user@xlogin0:~$ exit
salloc: Relinquishing job allocation 7731
user@xlogin0:~$
Always remember to exit your salloc subshell to release your resource allocation, otherwise you will continue to hold the allocation until the predefined time limit, and accure against your fairshare usage.

You can also directly use srun to obtain a job allocation to run a command directly:

user@xlogin0:~$ srun -N 2 hostname
xgpc0
xgpc1
user@xlogin0:~$
In the above, the specified argument means to allocate two tasks on distinct compute nodes. The difference between running salloc and invoking srun directly outside of an salloc allocation is that the former allows you to run multiple commands (i.e. steps) within the same allocation (i.e. the resources will not change), while the latter allocates and releases the allocation with each command invocation.

You can give the -p parameter to both salloc and srun to specify the partition to get your allocation from.

Both these methods of running jobs are not ideal, because you must wait for an allocation before the job will run.

There are many options that you can give to salloc and srun. Please refer to their man pages or the official Slurm documentation.

Job Management and Monitoring
squeue : Check the Slurm job queue. You can see jobs of other users in the queue as well.

scancel <jobid> : Cancel a specified job (jobid can be found when job is queued, or from the squeue command). If the job is running, it will be aborted.

sacct : List job accounting information. You can find jobs that are already over, including its status and exit code. This command can provide a lot of job and accounting information; please consult the man pages for more details.

sprio : List factors that comprises jobs' scheduling priority.

Slurm batch jobs output a *.log file. This contains the stdout of your batch script. If you need to check the “progress” of your batch script, you can check the output of this file. Ideally, you should setup your batch job to write informative output to a log file so that you can see how the job is progressing.

Interactive Shell
Although we generally say that you cannot get an interactive shell access into a compute node, there is actually a way to request one through Slurm. This may be useful to help debug problems that happen only while running under Slurm. Please note that any idle time is accrued against your fairshare usage and thus deprioritises your subsequent job allocations. Users who keep idling job allocations will also be penalised in their subsequent job allocation priority.

user@xlogin1:~$ salloc
salloc: Granted job allocation 7763
user@xlogin1:~$ srun --pty bash
user@xcnb0:~$ hostname
xcnb0
user@xcnb0:~$ exit
exit
user@xlogin1:~$ exit
exit
salloc: Relinquishing job allocation 7763
user@xlogin1:~$
Please do not hog resources by needlessly hanging on to them through such shell accesses. Time spent still counts towards your fairshare usage, i.e. you may be disadvantaged in future scheduling due to the amount of resources you've consumed.

Note: If you think that interactive shell access is the “only practical method” to use the cluster, you need to change your mindset and working style. A compute cluster is not like a standard shared server or your personal PC.

Selecting Nodes for Job
By default, Slurm will consider all cluster nodes for the scheduling of your job. Since the cluster comprises many different hardware types with varying configuraitons, this may sometimes be undesirable. For example, you may require to use a specific type of GPU like the NVIDIA Titan V. In this case, you can specify the type of GPU required by using the -G parameter.

$ salloc -G titanv
The list of valid GPU types can be found in slurm-info.

You can choose the exact compute node to run your job on if, for some reason, you absolutely require that very specific node.. You should not do this under most circumstances, since it is most disadvantageous to you when it comes to scheduling priority. Specific nodes can be selected by using the -w parameter with salloc or sbatch.

$ salloc -w xcna99
Given that the cluster is a shared environment, and you should setup your jobs to not have dependency on any on specific nodes. It is best that you select nodes by other criterion.

Note that your job will only be considered for scheduling when the required node is available.

Job Priority
The priority of jobs currently in queue is listed by the sprio command. Several factors go into determining a job's priority. The two most important factors are “fairshare”, and a combination of “partition” and “nice”. (Note: The nice value is shown only with the %N output format field.)

Please refer to job-priority for more information.



Node	Qty	OS	CPU	RAM	Storage	Network	GPU	Year Added	Brand
Slurm Compute-Only Nodes
xcna0 - xcna15	16	Ubuntu	2 x E5-2620 v2	32GB DDR3	4TB SATA HDD	1GbE	-	2014	Quanta
xcnb0 - xcnb19	20	Ubuntu	2 x Intel E5-2620 v2	64GB DDR3	4TB SATA HDD	1GbE	-	2015	Quanta
xcnc0 - xcnc49	50	Ubuntu	2 x Intel E5-2620 v3	64GB DDR4	4TB SATA HDD	1GbE	-	2015	Lenovo
xcnd0 - xcnd59	60	Ubuntu	2 x Intel E5-2620 v3	64GB DDR4	4TB SATA HDD	1GbE	-	2016	Huawei
xcne0 - xcne7	8	Ubuntu	2 x Intel Xeon Gold 6230 (2.1GHz)	768GB DDR4	960GB NVME & 1.92TB NVME	10GbE	-	2020	Asus
xcnf0 - xcnf25	26	Ubuntu	1 x AMD EPYC 7763	1TB DDR4	3.84TB NVME	10GbE		2023	Gigabyte
xcng0 - 1	2	Ubuntu	1 x Ampere Max M128-30	1TB DDR4	3.84TB NVME	10GbE		2023	Gigabyte
Slurm Compute with GPU Nodes
xgpa0 - xgpa4	5	Ubuntu	2 x Intel E5-2620 v2	32GB DDR3	1TB SAS HDD	1GbE	2x K20m GPU	2014	SuperMicro
xgpb0 - xgpb2	3	Ubuntu	2 x Intel E5-2620 v4	64GB DDR4	1.2TB SSD	10GbE	1x NVIDIA Tesla P4 GPU	2017	SuperMicro
xgpc0 - xgpc9	10	Ubuntu	2 x Intel Xeon Silver 4108	128GB DDR4	960GB SSD & 6TB SATA HDD	10GbE	1x NVIDIA Tesla V100 GPU	2018	Asus
xgpd0 - xgpd9	10	Ubuntu	2 x Intel Xeon Silver 4108	128GB DDR4	960GB SSD & 6TB SATA HDD	10GbE	2x NVIDIA Titan V GPU	2018	Asus
xgpe0 - xgpe9	10	Ubuntu	2 x Intel Xeon Silver 4116 (2.1Ghz)	256GB DDR4	960GB SSD & 3.84TB SSD	10GbE	1x NVIDIA Titan RTX GPU	2019	Asus
xgpe10	1	Ubuntu	2 x Intel Xeon Silver 4116 (2.1Ghz)	256GB DDR4	960GB SSD & 3.84TB SSD	10GbE	2x NVIDIA Titan RTX GPU	2019	Asus
xgpe11	1	Ubuntu	2 x Intel Xeon Silver 4116 (2.1Ghz)	512GB DDR4	960GB SSD & 3.84TB SSD	10GbE	2x NVIDIA Titan RTX GPU	2019	Asus
xgpf0 - xgpf9	10	Ubuntu	2 x Intel Xeon Silver 4116 (2.1Ghz)	256GB DDR4	960GB SSD & 3.84TB SSD	10GbE	1x NVIDIA Tesla T4 GPU	2019	Asus
xgpf10	1	Ubuntu	2 x Intel Xeon Silver 4116 (2.1Ghz)	256GB DDR4	960GB SSD & 3.84TB SSD	10GbE	2x NVIDIA Tesla T4 GPU	2019	Asus
xgpf11	1	Ubuntu	2 x Intel Xeon Silver 4116 (2.1Ghz)	512GB DDR4	960GB SSD & 3.84TB SSD	10GbE	2x NVIDIA Tesla T4 GPU	2019	Asus
xgpg0-xgpg9	10	Ubuntu	2x AMD Epyc 7352	256GB	960GB SSD & 3.8TB SSD	10GbE	1x NVIDIA A100 40GB	2021	Supermicro
amdgpu0-amdgpu3	4	Ubuntu	1x AMD EPYC 7642	512GB	960GB SSD	10GbE	8x AMD MI50	2022	Gigabyte
xgph0-xgph19	20	Ubuntu	2x Intel Xeon Gold 6326	256GB	3.84TB SSD	10GbE	1x NVIDIA A100 80GB configured as 2x 40GB MIG devices	2022	Supermicro
xgpi0-xgpi20	21	Ubuntu	2x AMD Epyc 9334	1TB	7.68TB SSD	10GbE	2x NVIDIA H100 96GB with 10 partitioned as 2x 47GB MIG devices and the rest native	2024	HPE
xgpj0	1	Ubuntu	1 x Ampere Altra	128GB	480GB SSD	10GbE	1x NVIDIA A100 80GB	2024	Gigabyte
xgpk0	1	Ubuntu	2x AMD EPYC 9355	1TB	1TB NVMe, 2x 14TB NVMe	10GbE	4x NVIDIA H100 141GB	2025	Supermicro
Slurm Login Nodes
xlogin0 -xlogin2	3	Ubuntu	2x AMD EPYC 7532	256GB	2x 1.8TB SSD	10GbE	-	2022	Gigabyte
Contributed Nodes from Research Grants (not managed by Slurm)
cgpa0 - cgpa3	4	Ubuntu	2x Intel Xeon Silver 4214	128GB	960GB SSD & 8TB HDD	1GbE	4x NVIDIA RTX2080	2020	Asus
cgpa4-10	7	Ubuntu	2x Intel Xeon Silver 4214R	128GB	960GB SSD & 8TB HDD	1GbE	4x NVIDIA A5000	2021	Asus
cgpb0	1	Ubuntu	2x Intel Xeon Silver 4210	256GB	3.2TB SSD	10GbE	2x NVIDIA Tesla V100 GPU	2020	HPE

GPU
SoC Compute Cluster has several types of GPUs available. This page provides information on how you can use these GPUs.

GPU Hardware
The following types of GPUs are currently available.

GPU Type	Quantity	Slurm GPU Name	Slurm Feature Name
NVIDIA Tesla V100	58	nv	cuda70, v100
NVIDIA Titan V	nv	cuda70, titanv
NVIDIA Titan RTX	nv	cuda75, titanrtx
NVIDIA Tesla T4	nv	cuda75 t4
NVIDIA A100 with 40GB	30	a100-40	cuda80, a100
NVIDIA A100 with 80GB	10	a100-80	cuda80, a100
NVIDIA H100 with 47GB	50	h100-47	cuda90, h100
NVIDIA H100 with 96GB	20	h100-96	cuda90, h100
NVIDIA H200	4	h200-141	cuda90, h200
AMD MI50	16	amd	mi50
The Slurm GPU Name is used to choose from categories of GPUs, and the Slurm Feature Name provides additional control of GPU selection within those categories.

Selecting GPUs
Apart from requesting GPUs, you also need to specify the GPU type required for your job. If you do not specify a type, then the 'nv' GPU type will be used. This means that any node with a 'nv' GPU type will be considered for your job.

The 'nv' GPU type covers all the NVIDIA GPUs apart from the newer A100 GPUs that are identified differently due to different quota limits. If you need to choose a specific CUDA level within the different families of 'nv' GPU, you can use the Slurm Feature Name. E.g. if you only want cuda 7.5 nodes, then you can use the command:

sbatch --gres=gpu:nv:1 -C cuda75
You cannot specify more than one GPU type for a job. If you want to consider more than one GPU type for your job, then you'll have to submit your job multiple times using different GPU types, then cancel the remanining jobs after the first one runs. You can do this by using the same job name for the multple jobs:

sbatch -J gpujob --gres=gpu:nv:1 job.sh
sbatch -J gpujob --gres=gpu:a100-40:1 job.sh
sbatch -J gpujob --gres=gpu:h100-47:1 job.sh
And then in the job.sh script, cancel all remaining jobs:

#!bin/sh

scancel -J gpujob --state=PENDING
Whichever job runs first, be it for 'nv', 'a100-40', or 'a100-80', will cancel the other jobs. (This is a simplistic example. Do consider a possible race condition where more than one job begins running before scancel gets the chance to cancel all other jobs. You may need to do additional checks.)