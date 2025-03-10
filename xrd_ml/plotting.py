from pandas import DataFrame
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

PLOTS_FOLDER = Path(__file__).parent.parent / "plots"

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


def plot_solid_fraction_distribution(processed_data: DataFrame,
                                     bins: int = 20,
                                     include_missing_hist = False) -> None:
    """
    Plot the solid fraction distribution present in avg_data.txt files of the dataset.

    Parameters:
        processed_data (DataFrame): DataFrame containing processed data
        include_missing_hist (bool): Whether to include bins with missing hist files
    """
    processed_data = processed_data.dropna(subset=["solidFrac"])

    if not include_missing_hist:
        # Drop rows where the xrd_data is None
        processed_data = processed_data.dropna(subset=["xrd_data"])

    processed_data.hist(column="solidFrac", bins=bins)

    plt.xlabel("Solid Fraction")
    plt.ylabel("Frequency")
    plt.title("Solid Fraction Distribution in Dataset")

def plot_model_predictions(true_solid_fractions, predicted_solid_fractions) -> None:
    """
    Plot the predicted solid fractions against the actual solid fractions.

    Parameters:
        true_solid_fractions (np.ndarray): True solid fractions
        predicted_solid_fractions (np.ndarray): Predicted solid fractions
    """
    plt.scatter(true_solid_fractions, predicted_solid_fractions, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # Perfect prediction line
    plt.xlabel('Actual Solid Fraction')
    plt.ylabel('Predicted Solid Fraction')
    plt.grid(True)

def plot_model_predictions_by_temp(true_solid_fractions, predicted_solid_fractions, temps) -> None:
    """
    Plot the predicted solid fractions against the actual solid fractions, colored by temperature.

    Parameters:
        true_solid_fractions (np.ndarray): True solid fractions
        predicted_solid_fractions (np.ndarray): Predicted solid fractions
        temps (np.ndarray): Temperature tuples (temp, melt_temp) in the same order as the solid fractions
    """
    temps = np.array(temps) # Convert to np array in case temps is list of tuples
    unique_temps = np.unique(temps, axis=0)
    unique_temps = [tuple(temp) for temp in unique_temps]
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_temps)))
    color_dict = {unique_temps[i]: colors[i] for i in range(len(unique_temps))}

    for temp_tuple in unique_temps:
        temp, melt_temp = temp_tuple
        mask = np.all(temps == np.array(temp_tuple), axis=1)
        plt.scatter(
            true_solid_fractions[mask],
            predicted_solid_fractions[mask],
            alpha=0.5,
            label=f"{temp} K, Melting Temp {melt_temp} K",
            color=color_dict[temp_tuple]
        )
        
    plt.plot([0, 1], [0, 1], 'r--')  # Perfect prediction line
    plt.xlabel('Actual Solid Fraction')
    plt.ylabel('Predicted Solid Fraction')
    plt.legend()


def save_plot(plot_name: str) -> None:
    """
    Save the current plot to the plots folder.

    Parameters:
        plot_name (str): Name of the plot file
    """
    PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_FOLDER / plot_name)
    plt.close()





