from plotting import plot_solid_fraction_distribution
import matplotlib.pyplot as plt
from load_data import load_processed_data

if __name__ == "__main__":

    print("Loading dataset...")
    train = load_processed_data(suppress_load_errors=True)

    print(f"Making plots...")
    plt.figure()
    plot_solid_fraction_distribution(train, bins=20, include_missing_hist=True)
    plt.title("Solid Fraction Distribution (with unprocessed XRDs)")
    plt.savefig("full_solid_fraction_distribution.png")
    print("Saved plot to full_solid_fraction_distribution.png")

    plt.figure()
    plot_solid_fraction_distribution(train, bins=20, include_missing_hist=True)
    plt.title("Solid Fraction Distribution (only processed XRDs)")
    plt.savefig("processed_solid_fraction_distribution.png")
    print("Saved plot to processed_solid_fraction_distribution.png")