import pandas as pd 
from pathlib import Path
import numpy as np
import re

PROCESSED_DATA_PATH = Path("/gpfs/sharedfs1/MD-XRD-ML/02_Processed-Data")
NUM_BINS_PER_TIMESTEP = 5

def load_avg_data(filepath: Path | str) -> pd.DataFrame:
    """
    Parse a file containing the specified data format into a pandas DataFrame.

    Example usage:
    df = parse_file_to_dataframe("path_to_your_file.txt")

    Parameters:
    filepath (str): Path to the input file containing the data.

    Returns:
    pd.DataFrame: A dataframe containing the parsed data.
    """
    # Read the content of the file
    with open(filepath, 'r') as file:
        data = file.read()
    
    # Split the input data into lines
    lines = data.strip().split("\n")
    
    # Extract the headers from the first line
    headers = lines[0].split(",")
    headers = [header.strip() for header in headers]
    
    # Initialize a list to store parsed rows
    rows = []
    
    # Process each subsequent line
    for line in lines[1:]:
        # Split the line by whitespace or tabs
        parts = line.split()
        if len(parts) > len(headers):
            # Handle the Bin # which might have multiple spaces
            parts = [parts[0]] + parts[1:]
        rows.append(parts)
    
    # Create a dataframe from the parsed data
    df = pd.DataFrame(rows, columns=headers)
    
    # Convert numerical columns to appropriate types
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            continue

    return df

def load_xrd_hist(filepath: Path | str) -> pd.DataFrame:
    """
    Load .hist.xrd file into DataFrame
    
    Parameters:
        filepath (str): Path to .hist.xrd file
    
    Returns:
        pd.DataFrame: DataFrame with XRD pattern data
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()

    if not lines:
        raise ValueError("The file is empty.")
    
    if len(lines) <= 4:
        raise ValueError("The file does not contain enough data.")
    
    data_start_idx = 4
    
    # Extract histogram data
    data = []
    for line in lines[data_start_idx:]:
        if line.strip():  # Ignore empty lines
            values = re.split(r'\s+', line.strip())
            bin_index = int(values[0])
            bin_coord = float(values[1])
            count = float(values[2])
            count_total = float(values[3])
            data.append([bin_index, bin_coord, count, count_total])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=["Bin", "Coord", "Count", "Count/Total"])
    return df
    
def load_processed_data(directory_path: Path | str = PROCESSED_DATA_PATH,
                        suppress_load_errors = False, 
                        verbose = False,
                        temps: list[int] | str = "all",
                        melt_temps: list[int] | str = "all"
                        ) -> pd.DataFrame:
    """
    Load all data from the processed data directory.
    
    Parameters:
        filepath (str): Path to the processed data file
        suppress_load_errors (bool): Whether to suppress errors during loading
        verbose (bool): Whether to print debug information
        
    Returns:
        pd.DataFrame: DataFrame containing the processed data

    The data is stored in a dataframe with columns:
        "temp": "int64",
        "melt_temp": "int64",
        "timestep": "object",
        "bin_num": "int64",
        "min Z": "float64",
        "max Z": "float64",
        "NAN bin": "int64",
        "avg T": "float64",
        "NAN other": "float64",
        "NAN FCC": "float64",
        "NAN HCP": "float64",
        "NAN BCC": "float64",
        "avg CSP": "float64",
        "xrd_data": "object" (pd Dataframe or None)
    """
    directory_path = Path(directory_path)

    if directory_path.is_file():
        raise ValueError("Please provide a directory path, not a file path.")
    elif not directory_path.exists():
        raise ValueError("The specified directory does not exist.")

    column_dtypes = {
        "temp": "int64",
        "melt_temp": "int64",
        "timestep": "int64",
        "bin_num": "int64",
        "min Z": "float64",
        "max Z": "float64",
        "NAN bin": "int64",
        "avg T": "float64",
        "NAN other": "float64",
        "NAN FCC": "float64",
        "NAN HCP": "float64",
        "NAN BCC": "float64",
        "avg CSP": "float64",
        "xrd_data": "object"
    }

    # Initialize the DataFrame
    processed_data = pd.DataFrame(columns=column_dtypes.keys())
    processed_data = processed_data.astype(column_dtypes)

    if verbose:
        print("Loading data from:")
        print(f"{directory_path}")

    # Iterate through temperature directories
    for temp_dir in directory_path.glob("*_*-Kelvin"):

        temp = int(temp_dir.name.split('_')[1].split('-')[0])
        if temps != "all" and temp not in temps:
            continue

        if verbose:
            print(f"\t{temp_dir.name}")

        for melt_dir in temp_dir.glob("*-Kelvin"):


            melt_temp = int(melt_dir.name.split('-')[0])
            if melt_temps != "all" and melt_temp not in melt_temps:
                continue

            if verbose:
                print(f"\t\t{melt_dir.name}")

            # Process each timestep
            for avg_file in melt_dir.glob("avg-data.*.txt"):
                timestep = avg_file.stem.split('.')[-1]

                # Load avg data
                avg_data = None
                if avg_file.exists():
                    try:
                        avg_data = load_avg_data(avg_file)
                    except Exception as e:
                        if not suppress_load_errors:
                            print(f"Error loading {avg_file}: {str(e)}")

                for bin_num in range(1, NUM_BINS_PER_TIMESTEP + 1):

                    # Load corresponding XRD data
                    xrd_file = melt_dir / f"{timestep}.{bin_num}.hist.xrd"
                    xrd_data = None
                    if xrd_file.exists():
                        try:
                            xrd_data = load_xrd_hist(xrd_file)
                        except Exception as e:
                            if not suppress_load_errors:
                                print(f"Error loading {xrd_file}: {str(e)}")
                        
                    

                    # Append the row to DataFrame
                    new_row = pd.DataFrame([{
                        "temp": temp,
                        "melt_temp": melt_temp,
                        "timestep": timestep,
                        "bin_num": bin_num,
                        "min Z": avg_data["min Z"].iloc[bin_num - 1] if avg_data is not None else np.nan,
                        "max Z": avg_data["max Z"].iloc[bin_num - 1] if avg_data is not None else np.nan,
                        "NAN bin": avg_data["NAN bin"].iloc[bin_num - 1] if avg_data is not None else np.nan,
                        "avg T": avg_data["avg T"].iloc[bin_num - 1] if avg_data is not None else np.nan,
                        "NAN other": avg_data["NAN other"].iloc[bin_num - 1] if avg_data is not None else np.nan,
                        "NAN FCC": avg_data["NAN FCC"].iloc[bin_num - 1] if avg_data is not None else np.nan,
                        "NAN HCP": avg_data["NAN HCP"].iloc[bin_num - 1] if avg_data is not None else np.nan,
                        "NAN BCC": avg_data["NAN BCC"].iloc[bin_num - 1] if avg_data is not None else np.nan,
                        "avg CSP": avg_data["avg CSP"].iloc[bin_num - 1] if avg_data is not None else np.nan,
                        "xrd_data": xrd_data
                    }])
                    new_row = new_row.astype(column_dtypes)
              

                    processed_data = pd.concat([processed_data, new_row], ignore_index=True)

    processed_data = processed_data.sort_values(by=["temp", "melt_temp", "timestep", "bin_num"])

    return processed_data   

def get_usable_bins(processed_data: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out rows with missing avg_data or xrd_data.
    """
    return processed_data[(processed_data["avg T"].notna()) & (processed_data["xrd_data"].notna())]

def get_missing_bins(processed_data: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the rows with missing avg_data or xrd_data.
    """
    return processed_data[(processed_data["avg T"].isna()) | (processed_data["xrd_data"].isna())]

def get_entirely_missing_timesteps(processed_data: pd.DataFrame, temp: int, melt_temp: int) -> list[int]:
    """
    Return the timesteps that have missing histograms for all bins.
    """
    missing_bins = get_missing_bins(processed_data)

    # get rows with specified temp and melt_temp
    missing_bins = missing_bins[(missing_bins["temp"] == temp) & (missing_bins["melt_temp"] == melt_temp)]

    # get timesteps with missing histograms for all bins
    # there should be num_bins_per_timestep rows for each timestep
    missing_timesteps = missing_bins.groupby("timestep").filter(lambda x: x.shape[0] == NUM_BINS_PER_TIMESTEP)["timestep"].unique()

    return missing_timesteps.tolist()

if __name__ == "__main__":
    path = Path("/gpfs/sharedfs1/MD-XRD-ML/02_Processed-Data")

    # Load a single avg data file
    avg_data = load_avg_data(path / "01_300-Kelvin" / "2500-Kelvin" / "avg-data.0.txt")
    print("Avg Data Example:")
    print(avg_data.head())
    print("")

    # Load a single xrd hist file
    xrd_data = load_xrd_hist(path / "01_300-Kelvin" / "2500-Kelvin" / "0.1.hist.xrd")
    print("XRD Data Example:")
    print(xrd_data.head())
    print("")
    
    # Load all the data
    processed_data = load_processed_data(path, verbose=True)
    print(processed_data.head())

    # Print out the rows that are missing avg_data or xrd_data
    missing_bins = get_missing_bins(processed_data)
    if not missing_bins.empty:
        print(missing_bins[["temp", "melt_temp", "timestep", "bin_num"]].to_string(index=False))
    else:
        print("No rows found with missing avg_data or xrd_data.")

    # Count the number of samples (bins) with usable data
    usable_bins = get_usable_bins(processed_data)
    print(f"Number of usable bins: {usable_bins.shape[0]}")
