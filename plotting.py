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
    plt.xlabel("2$\Theta$")
    plt.ylabel("Count")

    if title:
        plt.title(title)
    else:
        plt.title("XRD Histogram")


