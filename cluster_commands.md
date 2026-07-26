# <center> AMD cluster cheat sheet </center>

<font color="lightblue">

### <font color="orange"> Basics: </font>

#### <font color="silver"> Nodes: </font>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Retrieve information about resources available: `sinfo`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <font color="pink"> Get information on one node: `scontrol show node NODENAME` </font>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Allocate a node: `salloc -p PARTITION --time=HH:MM:SS`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <font color="pink"> View allocated nodes or submitted jobs: `squeue -u arauch` </font>

#### <font color="silver"> Modules: </font>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Show loadable modules: `modules avail`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Allow loading modules defined in another path: `module use MODULE_PATH`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <font color="pink"> Display modules already loaded: `module list` </font>

#### <font color="silver"> Virtual environments: </font>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Create: `python3 -m venv qaoa_env` \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Enter: `source ~/qaoa_env/bin/activate` \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Leave: `deactivate`

#### <font color="silver"> Import files: </font>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Import from Github into local: `git clone https://github.com/alban-rauch/qaoa-mis.git` \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Copy from local into cluster: `scp -r qaoa-mis arauch@AMDmuncl03:~`

### <font color="orange"> Submitting a job: </font>

#### <font color="yellow"> Interactive run: </font>
```
salloc -p PARTITION --time=HH:MM:SS
module load python/3.12.13.rocm
source ~/qaoa_env/bin/activate
srun python3 scripts/test.py
```

#### <font color="yellow"> Batch job: </font>
```
sbatch SLURM_SCRIPT
```

### <font color="orange"> Slurm script: </font>

```
#!/bin/bash
################################################################################################
#SBATCH -J quantum_sim              # Job name               (or --job-name=quantum_sim)
#SBATCH -p MI350P_600W             # Partition name         (or --partition=0745-1R5600-NOIB)
#SBATCH -N 1                        # Nodes requested        (or --nodes=1)
#SBATCH --exclusive                 # Resource not shared with other users
#SBATCH --time=00:10:00             # Time limit
#SBATCH --output=outputs/%x_%j.out  # Output file
#SBATCH --error=outputs/%x_%j.err   # Error file
################################################################################################

module purge
module load python/3.12.13.rocm

source ~/qaoa_env/bin/activate

cd ~/test

python3 scripts/test.py
```

Add if necessary: \
`srun` \
`#SBATCH --ntasks-per-node=4` \
`#SBATCH --gpus-per-node=4`


