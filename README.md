# Building Machine Learning Models to Interpret Virtual X-ray Diffraction Patterns​ - UConn Computer Science and Engineering Senior Design Team 15

For a project overview, see https://seniordesignday.engr.uconn.edu/seniorprojectpt/senior-design-2025-computer-science-and-engineering-team-15/

## Directory Structure
- Put Python modules in folder `xrd_ml`
- Put sbatch scripts in folder `hpc_jobs`

## Python Environment Setup
1. Follow instrcuctions here to setup your Python environment on the UConn HPC using miniconda.
     https://kb.uconn.edu/space/SH/26079723879/Miniconda+Environment+Set+Up
2. Inside of the repository folder, run `conda env create -f environment.yml -n xrd_ml` to automatically create a conda environment called xrd_ml
3. Activate the environment with `conda activate xrd_ml`
4. When you create a job `.sh` file, include the line `conda activate xrd_ml` 
5. If you install more packages, update the environment.yml file by running `conda env export > environment.yml`

## Code Documentation

All scripts can be run with the `-h` argument to see their description and arguments.

### Model Training Scripts

#### Support Vector Regression (SVR)
- `svr.py`
  - Trains an SVR with hyperparameters specified with command line
  - Arguments:
     - `C`
     - `gamma`
     - `epsilon`
     - `test`
          - If provided, train on training set + validation set and report results on test set.
          - If not provided, train on training set and and report results on validation set.
          - Currently, this script does not support selecting the `train_test_split`. It only uses our original split.
- `svr_gridsearch.py`
  - *This is what we used to generate our final SVR results*
  - Trains SVRs on the training set without various hyperparameter combinations
  - The SVR which performs best is on the validation set is then used for evaluation on the test set
- `svr_gridsearch.py`
     - Uses scikit learn GridSearchCV (grid search with cross validation) to train on combined training + validation set
     - The hyperparameters which perform best are used for evaluation on the test set

#### Convolutional Neural Network
TODO

#### Random Forest
TODO

#### Gradient Boosting Machines
TODO

### Utility Scripts
- `plot_solid_fraction_distribution.py`
     - Plots the solid fraction distribution for the entire dataset, as well as for the the train, validation and test data.
     - Plots are saved to `xrd-ml/plots` in a timestamped subdirectory
     - Has `train_test_split` and `balance` arguments
- `submit_xrd_jobs.py`
     - Script we used to automate submission of jobs to run XRD on .bin files
     - Can detect missing timesteps and automatically submit jobs by temporarily modifying and executing the `Job-Submission-Individual-Files-V1.0.sh` file which is stored along with our data on the HPC

All scripts can be run with the `-h` parameter to see their description and arguments.


### Python Helper Modules
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
     - Defines the PLOTS_FOLDER, which is used to store all plots (set to `plots` at root level of repository)








