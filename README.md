# sdp

## Directory Structure
- Put Python modules in folder `xrd_ml`
- Put sbatch scripts in folder `hpc_jobs`

## Python Environment Setup
1. Follow instrcuctions here to setup your Python environment on the hpc using miniconda.
     https://kb.uconn.edu/space/SH/26079723879/Miniconda+Environment+Set+Up
2. Inside of the repository folder, run `conda env create -f environment.yml` to automatically create a conda environment called xrd_ml
3. Activate the environment with `conda activate xrd_ml`
4. If you create more jobs, include the line `conda activate xrd_ml` 
5. If you install more packages, update the environment.yml file by running `conda env export > environment.yml`






