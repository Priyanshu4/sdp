import tempfile
import subprocess
import os
from pathlib import Path
from pandas import DataFrame
from load_data import (
    PROCESSED_DATA_PATH,
    TEMP_TO_DIRECTORY,
    MELTING_TEMP_TO_DIRECTORY,
    get_entirely_missing_timesteps, 
    get_missing_bins,
    load_processed_data_for_temp_directory,
)


def run_job_submission_individual_files(directory: Path | str, file_start: int, file_stop: int):
    """
    Run job submission for individual files in the given directory.

    Parameters:
        directory (str): Path to the directory containing the files
        file_start (int): Starting file number
        file_stop (int): Ending file number
    """
    directory = Path(directory)
    
    job_submission_individual_files_sh_path = directory / "Job-Submission-Individual-Files-V1.0.sh"

    if not job_submission_individual_files_sh_path.exists():
        raise FileNotFoundError("Job submission script not found.")

    # Read the original script content
    with open(job_submission_individual_files_sh_path, "r") as file:
        script_content = file.readlines()

    # Modify fileStart and fileStop
    for i, line in enumerate(script_content):
        if line.strip().startswith("fileStart="):
            script_content[i] = f"fileStart={file_start}\n"
        elif line.strip().startswith("fileStop="):
            script_content[i] = f"fileStop={file_stop}\n"

    # Create a temporary script in the same directory as the original script
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", dir=directory) as temp_script:
        temp_script_path = temp_script.name  # Get the temporary file path
        temp_script.writelines(script_content)
        temp_script.flush()
        os.chmod(temp_script_path, 0o755)
        subprocess.run(["bash", temp_script_path], cwd=directory, check=True)


def submit_xrd_job(temp: int, melting_temp: int, timestep: int):
    """
    Submit an XRD job for the given temperature, melting temperature, and timestep.

    Parameters:
        temp (int): Temperature
        melting_temp (int): Melting temperature
        timestep (int): Timestep
    """
    directory = PROCESSED_DATA_PATH / TEMP_TO_DIRECTORY[temp] / MELTING_TEMP_TO_DIRECTORY[melting_temp]

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    run_job_submission_individual_files(directory, timestep, timestep)
  

def submit_xrd_jobs_for_entirely_missing_timesteps(data: DataFrame, temp: int, melting_temp: int):
    """
    Submit XRD jobs for timesteps where all bins are missing histograms.

    Parameters:
        data (DataFrame): Processed data
        temp (int): Temperature
        melting_temp (int): Melting temperature
    """
    
    # find the timesteps that have missing histograms for all bins
    missing_timesteps = get_entirely_missing_timesteps(data, temp, melting_temp)

    for timestep in missing_timesteps:
        submit_xrd_job(temp, melting_temp, timestep)


if __name__ == "__main__":

    print("Options:")
    print("1. Submit XRD job for a specific timestep")
    print("2. Submit XRD jobs for entirely missing timesteps in a directory")
    print("3. Get missing bins for all timesteps in a directory")

    option = int(input("Enter option: "))

    if option == 1:
        print("Submitting job for one timestep...")
        temp = int(input("Enter temperature: "))
        melting_temp = int(input("Enter melting temperature: "))
        timestep = int(input("Enter timestep: "))

        try:
            submit_xrd_job(temp, melting_temp, timestep)
        except Exception as e:
            print("Job not submitted. Inputs are invalid.")
            print(e)

    elif option == 2:
        print("Submitting jobs for entirely missing timesteps in a directory...")
        temp = int(input("Enter temperature: "))
        melting_temp = int(input("Enter melting temperature: "))

        print(f"Loading processed data for temp {temp} and melting temp {melting_temp}...")
        processed_data = load_processed_data_for_temp_directory(temp, melting_temp, suppress_load_errors=True)

        missing_timesteps = get_entirely_missing_timesteps(processed_data, temp, melting_temp)

        if len(missing_timesteps) == 0:
            print("No timesteps were found with missing histograms for all 5 bins.")
            print("There may still be some timesteps with partially missing bins. Use option 3 to get missing bins.")
            exit()

        print("Jobs will be submitted for the following timesteps. These timesteps should be missing XRD data for every bin:")
        print(missing_timesteps)

        if input("Do you want to continue? (y/n) ") == "y":           
               for timestep in missing_timesteps:
                submit_xrd_job(temp, melting_temp, timestep)
        else:
            print("Jobs not submitted.")

    elif option == 3:

        print("Getting missing bins for all timesteps in a directory...")
        temp = int(input("Enter temperature: "))
        melting_temp = int(input("Enter melting temperature: "))

        print(f"Loading processed data for temp {temp} and melting temp {melting_temp}...")
        processed_data = load_processed_data_for_temp_directory(temp, melting_temp, suppress_load_errors=True)

        missing_bins_data = get_missing_bins(processed_data)

        if len(missing_bins_data) == 0:
            print("No missing bins were found. Double check your inputs.")
            exit()

        # create a dict mapping timestep to missing bins
        missing = dict()

        # iterate over rows of the missing bins data
        for i, row in missing_bins_data.iterrows():
            timestep = row["timestep"]
            bin = row["bin_num"]

            bin_list = missing.get(timestep, [])
            bin_list.append(bin)
            missing[timestep] = bin_list

        print("Missing bins for each timestep:")
        for timestep, bins in missing.items():
            print(f"Timestep {timestep}: {bins}")

        print()
        if input("Would you like to submit jobs for any of these timesteps? (y/n) ") != "y":
            print("Exiting.")
            exit()

        print("For any timesteps selected, all bins have to be rerun (even if some bins are already present).")
        
        for timestep, bins in missing.items():

            if input(f"Submit for timestep {timestep} with missing bins {bins}? (y/n) ") == "y":
                submit_xrd_job(temp, melting_temp, timestep)
                print(f"Job submitted for timestep {timestep}.")
            else:
                print(f"Job not submitted for timestep {timestep}.")
    else:
        print("Invalid option. Exiting.")

