from pandas import DataFrame
import matplotlib.pyplot as plt

def plot_xrd_hist(xrd_hist: DataFrame, title = "") -> None:
    """
    Plot XRD histogram from the given DataFrame.
    Does not call plt.savefig or plt.show, so the user should call these functions after this function.
    
    Parameters:
        xrd_hist (DataFrame): DataFrame containing XRD histogram data
    """    
    plt.figure()
    plt.plot(xrd_hist["Coord"], xrd_hist["Count"])
    plt.xlabel(r"2$\Theta$")
    plt.ylabel("Count")

    if title:
        plt.title(title)
    else:
        plt.title("XRD Histogram")

def plot_avg_temps_in_dataset(processed_data: DataFrame,
                              bins: int = 20,
                              include_missing_hist = False) -> None:
    """
    Plot the average temperatures present in avg_data.txt files of the dataset.

    Parameters:
        processed_data (DataFrame): DataFrame containing processed data
        include_missing_hist (bool): Whether to include bins with missing hist files
    """
    processed_data = processed_data.dropna(subset=["avg T"])

    if not include_missing_hist:
        # Drop rows where the xrd_data is None
        processed_data = processed_data.dropna(subset=["xrd_data"])

    processed_data.hist(column="avg T", bins=bins)

    plt.xlabel("avg T (K)")
    plt.ylabel("Frequency")
    plt.title("Average Temperatures in Dataset")





