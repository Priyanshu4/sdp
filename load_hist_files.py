import pandas as pd
import numpy as np
from pathlib import Path

def load_hist_xrd(filepath):
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

def process_all_data(root_dir):
    """
    Process all data files in directory structure
    
    Parameters:
        root_dir (str): Root directory containing data
    
    Returns:
        dict: Dictionary of processed DataFrames
    """
    root = Path(root_dir)
    processed_data = {}
    
    # Walk through directory structure
    for temp_dir in root.glob("*_*-Kelvin"):
        temp = int(temp_dir.name.split('_')[1].split('-')[0])
        
        for melt_dir in temp_dir.glob("*-Kelvin"):
            melt_temp = int(melt_dir.name.split('-')[0])
            
            # Process each timestep
            for avg_file in melt_dir.glob("avg-data.*.txt"):
                timestep = avg_file.stem.split('.')[-1]
                
                # Load avg data
                avg_data = parse_bin_avg_data(str(avg_file))
                
                # Load corresponding XRD data
                for bin_num in range(1, 6):
                    xrd_file = melt_dir / f"{timestep}.{bin_num}.hist.xrd"
                    if xrd_file.exists():
                        xrd_data = load_hist_xrd(str(xrd_file))
                        
                        # Store processed data
                        key = (temp, melt_temp, timestep, bin_num)
                        processed_data[key] = {
                            'avg_data': avg_data,
                            'xrd_data': xrd_data
                        }
                        
    return processed_data

# Example usage:
if __name__ == "__main__":
    root_dir = "/path/to/data"
    processed_data = process_all_data(root_dir)
