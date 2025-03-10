from pathlib import Path
import matplotlib.pyplot as plt
from load_data import load_processed_data
import datetime

if __name__ == "__main__":
    path = Path("/gpfs/sharedfs1/MD-XRD-ML/02_Processed-Data")

    
    # Load all the data
    print("Loading all the processed data...")
    processed_data = load_processed_data(path, verbose=False)
    print(processed_data.head())

    # pickeling the processed data, add date and time to the file name
    processed_data.to_pickle(f"processed_data_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl")


