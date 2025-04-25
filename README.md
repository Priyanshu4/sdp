# sdp

## Directory Structure
- Put Python modules in folder `xrd_ml`
- Put sbatch scripts in folder `hpc_jobs`

## Python Environment Setup
1. Follow instrcuctions here to setup your Python environment on the hpc using miniconda.
     https://kb.uconn.edu/space/SH/26079723879/Miniconda+Environment+Set+Up
2. Inside of the repository folder, run `conda env create -f environment.yml -n xrd_ml` to automatically create a conda environment called xrd_ml
3. Activate the environment with `conda activate xrd_ml`
4. When you create a job `.sh` file, include the line `conda activate xrd_ml` 
5. If you install more packages, update the environment.yml file by running `conda env export > environment.yml`

## Modules
These files are not meant to be executed directly, but provide helper functions that are used by our model training and utility scripts.

- `load_data.py`
     - Provides functions to load data from average data files and .hist.xrd files
     - If you would like to load data from a different location than what we have been using, then you would need to update and use the   functions provided by this file
- `train_test_split.py`
     - Provides functions to split and load train data, validation data and test data
     - Defines multiple different training and test splits and includes a dictionary mapping strings to each split
          - Other files which have a `split` or `train_test_split` argument use this dictionary
- `imbalance.py`
     - Provides functions to resample the dataset in order to handle data imbalance
     - `resample_dataset_from_binned_solid_fractions` is the function we ended up using (but not with the default arguments listed there, refer to our model files for the arguments we actually use)
- `plotting.py`
     - Provides functions to create plots of XRD histograms, data distributions and model results







