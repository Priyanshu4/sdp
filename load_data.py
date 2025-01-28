import pandas as pd 
from pathlib import Path

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

def load_hist_xrd(filepath: Path | str) -> pd.DataFrame:
    """
    Load .hist.xrd file into DataFrame
    
    Parameters:
        filepath (str): Path to .hist.xrd file
    
    Returns:
        pd.DataFrame: DataFrame with XRD pattern data
    """
    try:
        # Read the XRD data
        data = pd.read_csv(filepath, 
                          delim_whitespace=True,
                          comment='#',
                          header=None)
        
        # Rename columns appropriately 
        data.columns = ['2theta', 'intensity']
        
        return data
        
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None
    
def load_processed_data(directory_path: Path | str, verbose = False) -> pd.DataFrame:
    """
    Load all data from the processed data directory.
    
    Parameters:
        filepath (str): Path to the processed data file
        verbose (bool): Whether to print debug information
        
    Returns:
        pd.DataFrame: DataFrame containing the processed data

    The data is stored in a dataframe with columns:
    - temp
    - melt_temp
    - timestep
    - bin_num
    - avg_data (DataFrame or None)
    - xrd_data (DataFrame or None)
    """

    if Path(directory_path).is_file():
        raise ValueError("Please provide a directory path, not a file path.")
    elif not Path(directory_path).exists():
        raise ValueError("The specified directory does not exist.")

    # Initialize an empty DataFrame
    columns = ["temp", "melt_temp", "timestep", "bin_num", "avg_data", "xrd_data"]
    processed_data = pd.DataFrame(columns=columns)

    if verbose:
        print("Loading data from:")
        print(f"{directory_path}")

    # Iterate through temperature directories
    for temp_dir in directory_path.glob("*_*-Kelvin"):

        if verbose:
            print(f"\t{temp_dir.name}")

        temp = int(temp_dir.name.split('_')[1].split('-')[0])

        for melt_dir in temp_dir.glob("*-Kelvin"):

            if verbose:
                print(f"\t\t{melt_dir.name}")

            melt_temp = int(melt_dir.name.split('-')[0])

            # Process each timestep
            for avg_file in melt_dir.glob("avg-data.*.txt"):
                timestep = avg_file.stem.split('.')[-1]

                # Load avg data
                avg_data = None
                if avg_file.exists():
                    try:
                        avg_data = load_avg_data(avg_file)
                    except Exception as e:
                        print(f"Error loading {avg_file}: {str(e)}")

                # Load corresponding XRD data
                for bin_num in range(1, 6):
                    xrd_file = melt_dir / f"{timestep}.{bin_num}.hist.xrd"
                    
                    xrd_data = None
                    if xrd_file.exists():
                        try:
                            xrd_data = load_hist_xrd(xrd_file)
                        except Exception as e:
                            print(f"Error loading {xrd_file}: {str(e)}")

                    # Append the row to DataFrame
                    new_row = pd.DataFrame([{
                        "temp": temp,
                        "melt_temp": melt_temp,
                        "timestep": timestep,
                        "bin_num": bin_num,
                        "avg_data": avg_data,  # Store DataFrame as object
                        "xrd_data": xrd_data   # Store DataFrame as object
                    }])
                    
                    processed_data = pd.concat([processed_data, new_row], ignore_index=True)

    # Ensure 'avg_data' and 'xrd_data' are treated as objects
    processed_data["avg_data"] = processed_data["avg_data"].astype(object)
    processed_data["xrd_data"] = processed_data["xrd_data"].astype(object)

    # Set an index for optimized lookups
    processed_data.set_index(["temp", "melt_temp", "timestep", "bin_num"], inplace=True)

    return processed_data   

if __name__ == "__main__":
    path = Path("/gpfs/sharedfs1/MD-XRD-ML/02_Processed-Data")
    processed_data = load_processed_data(path, verbose=True)

    print(processed_data.head())
