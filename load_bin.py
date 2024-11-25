import pandas as pd 

def parse_bin_avg_data(filepath):
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
            df[column] = pd.to_numeric(df[column], errors='ignore')
        except ValueError:
            continue

    return df

