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

If any scripts produce plots, they will automatically create a `plots` directory within the repository and a subdirectory in the `plots` folder that is labeled with the model/script name and a timestamp. All of the plots for the script will be stored within this subdirectory. This is handled by the `set_plots_subdirectory` and `save_plot` functions in `plotting.py`.

### Model Training Scripts
All model training scripts share the following arguments. The arguments listed for each model script are in addition to these:
  - `--train_test_split`: Selects which train test split to use. List the options with `-h` or see the `TRAIN_TEST_SPLIT` dictionary in `train_test_split.py`
     - The split used to generate our final results is `train_2000_val_2500_test_3500`, which is the default for all our model scripts
  - `--balance`: Use resampling to balance the training data distribution with undersampling.
    - Balancing was only tested with the `train_2000_val_2500_test_3500` split, and the parameters used in the balancing function may not be appropriate for other splits.

#### Support Vector Regression (SVR)
- `svr.py`
  - Trains an SVR with hyperparameters specified with command line
  - Arguments:
     - `--C`
     - `--gamma`
     - `--epsilon`
     - `--test`
          - If provided, train on training set + validation set and report results on test set.
          - If not provided, train on training set and and report results on validation set.
          - Currently, this script does not support selecting the `train_test_split`. It only uses our original split.
- `svr_gridsearch.py`
  - ***This is what we used to generate our final SVR results***
  - Trains SVRs on the training set with gridsearch for hyperparameter optimization
  - Gridsearch tunes the `C`, `gamma` and `epsilon`
  - The SVR which performs best on the validation set is then used for evaluation on the test set
- `svr_gridsearch.py`
     - Uses scikit learn GridSearchCV (grid search with cross validation) to train on combined training + validation set
     - The hyperparameters which perform best are used for evaluation on the test set

#### Convolutional Neural Network
TODO

#### Random Forest
- `rf_gridsearch.py`
  - ***This is what we used to generate our final Random Forest results***
  - Trains a Random Forest model with grid search for hyperparameter optimization
  - Gridsearch tunes the `n_estimators` (Number of trees in the forest), `max_depth` (Maximum depth of trees) and `min_samples_split` (Minimum samples required to split a node) 
  - The random forest which performs best on the validation set is then used for evaluation on the test set

#### Gradient Boosting Machines
- `gbm.py`
  - ***This is what we used to generate our final Gradient Boosted Machine results***
  - Trains an XGBoost model for regression on XRD data
  - Arguments:
     - `--mode`: Select "validation" or "test" mode
     - `--lr`: Set the learning rate (default: 0.01)
     - `--depth`: Set the maximum tree depth (default: 6)
     - `--boost-rounds`: Set the number of boosting rounds (default: 1000)
     - `--tune`: Perform hyperparameter tuning with GridSearchCV
     - `--quick-tune`: Use a smaller parameter grid for faster tuning

### Utility Scripts
- `plot_solid_fraction_distribution.py`
     - Plots the solid fraction distribution for the entire dataset, as well as for the the train, validation and test data.
     - Plots are saved to `xrd-ml/plots` in a timestamped subdirectory
     - Has `split` and `balance` arguments
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








