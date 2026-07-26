# Cheat sheet for commands in bash and code in Python / Slurm

## Important


### (1) Upload files

`scp -r QAOA_GPU AMDmuncl03:/home/arauch`
`git clone https://github.com/alban-rauch/QAOA.git`

### (2) Connect to cluster

`ssh AMDmuncl03`


### (3) Submit job

`cd QAOA_QPU`


For little experiments/debug:

`salloc -p MI300A --gres=gpu:1 -t 00:30:00 -J qaoa_test`

`module load rocm python`

`python scripts/test.py`


For larger experiments:

`sbatch --job-name=test slurm/submit_job.sh test.py`
- Run test from "scripts/" folder with given job name
- Puts output files in folder "outputs/"
- To submit plots in folder "plots/", write plt.savefig("../plots/quantum_plot.png")

`squeue -u arauch`
- Check queuing

`cat outputs/test_JOBID.out`
- Read outputs


### (4) Download results

`scp -r AMDmuncl03:/home/arauch/QAOA_GPU/outputs`


## Build process (C, C++)

1. C++ Code + CMakeLists.txt
    * **CMake**
2. Build files (Makefile)
    * **Make/Ninja**
3. Compiler (GCC)
    * **Compilation**
4. Executable

### CMake
1. Scans to find compiler
2. Checks where libraries are installed
3. Detects operating system
- Generates build scripts (converts CMakeLists.txt into system's build script format) -> Makefile

### Build tool
- When running cmake --build or make etc.
1. Dependency tracking: Looks at code files (.cpp) and header (.h) to determine dependencies
2. Incremental building: only compile newly edited files
3. Parallel processing: ahnds tasks off to CPU cores simultaneously so multiple code files compile together
4. Invocation: feeds individual files to the compiler

### Compilation
1. Compiler: takes source code files (.cpp) and converts them into machine code object files (.o)
    - Checks for syntax errors, optimizes performance, and translates human-readable C++ directly into binary instructions (0s and 1s) tailored to your CPU architecture.
2. Linker: takes separate .o files, and external libraries, and stitches them together
    - Resolves cross-references

### Executable
- .exe file

## Slurm

Available partitions: `sinfo`

`sinfo -o "%P %G %D %T"`
- %P: Partition
- %G: GRES (gpu etc.)
- %D: Node count
- %T: State ("idle": zero jobs running | "drained": unavailable to accept any new jobs)

Check queue: `squeue  -u $USER`

Detailed information: `scontrol show job JOBID`

Available nodes: `scontrol show nodes`

Cancel job: `scancel JOBID`

Finished jobs: `sacct`


## Modules

Available modules: `module avail`

Load module: `module load rocm python`

Loded modules: `module list`

Unload everything: `module purge`

`module load rocm/(version1) python/(version2)`
- "rocm" = Radeon Open Compute (talk to the GPUs)
- "python" = cluster's official version of Python
- Examples:   version1 = 6.1.0, version2 = 3.11
- If default: remove version1, version2




## Slurm script

`#SBATCH --job-name=quantum_sim`       (Job name) \
`#SBATCH --partition=MI300A` (Partitions are departments, like "cpu", "gpu", "debug" [Here: we want computers with gpu]) \
`#SBATCH --nodes=1` (Nodes are computers in the server rack [Here: isolate 1 node]) \
`#SBATCH --gres=gpu:1` (GRES = Generic Resources (eg GPU, not standard CPU / RAM) [Here: request 1 AMD GPU (MI350P / Radeon 9700)]) \
`#SBATCH --time=00:10:00` (Time limit (HH:MM:SS) - e.g., 10 mins) \
`#SBATCH --output=outputs/%x_%j.out` (Output file (where prints go)) \
`#SBATCH --error=outputs/%x_%j.err` (Error log file)


Useful Slurm tokens:
- `%x` = job name
- `%j` = job ID in the cluster
- `%u` = my cluster username
- `%N` = node/computer on which the job was run

`module load rocm python`

`python scripts/$1`



## Bash

Generate ssh key: `ssh-keygen`

Print public key (found in document ".ssh/id_ed25519.pub"): `more .ssh/id_ed25519.pub` / `cat .ssh/id_ed25519.pub`

Edit the config file: `vi .ssh/config`
* "i" for insert mode
* "Esc" to quit insert mode
* ":wq" to save and exit
* ":q!" to throw away and exit

View the config file: `cat .ssh/config`

Secure and lock access: `chmod 600 .ssh/config`
* People: (Owner | Group | Others)
* Permissions: 4 = read, 2 = write, 1 = execute

See VPN host: `host vpn-labos.polytechnique.fr`

See pre-installed software: `module avail`

Run file from GitHub: `curl -s "https://raw.githubusercontent.com/alban-rauch/QAOA/refs/heads/main/Test%20scripts%20for%20AMD%20GPU/scripts/rotator.py" | python3`


## Python scripts

`dev = qp.device("device", wires=1)`

Devices:
* "default.qubit"    (vanilla)
* "qulacs.simulator" (qulacs)
* "lightning.qubit"  (CPU)
* "lightning.amdgpu" (AMD GPU)
