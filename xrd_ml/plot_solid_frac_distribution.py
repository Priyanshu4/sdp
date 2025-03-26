from plotting import plot_solid_fraction_distribution, save_plot
import matplotlib.pyplot as plt
from load_data import (
    load_processed_data,
    get_usable_bins
)
from train_test_split import (
    load_train_data,
    load_validation_data,
    load_test_data
)

if __name__ == "__main__":

    print("Loading dataset...")
    full_dataset = load_processed_data(suppress_load_errors=True)

    print(f"Making plots...")
    plt.figure()
    plot_solid_fraction_distribution(full_dataset, bins=20, include_missing_hist=True)
    plt.title("Solid Fraction Distribution (with unprocessed XRDs)")
    save_plot("full_solid_fraction_distribution.png")
    print("Saved plot to full_solid_fraction_distribution.png")

    plt.figure()
    plot_solid_fraction_distribution(full_dataset, bins=20, include_missing_hist=False)
    plt.title("Solid Fraction Distribution (only processed XRDs)")
    save_plot("processed_solid_fraction_distribution.png")
    print("Saved plot to processed_solid_fraction_distribution.png")

    print("Loading training data...")
    train = load_train_data(suppress_load_errors=True)
    print("Number of total training bins: ", len(train))
    print("Number of usable training bins: ", len(get_usable_bins(train)))
    plt.figure()
    plot_solid_fraction_distribution(train, bins=20, include_missing_hist=False)
    plt.title("Training Data: Solid Fraction Distribution")
    save_plot("train_solid_fraction_distribution.png")
    print("Saved plot to train_solid_fraction_distribution.png")

    print("Loading validation data...")
    validation = load_validation_data(suppress_load_errors=True)
    plt.figure()
    print("Number of total validation bins: ", len(validation))
    print("Number of usable validation bins: ", len(get_usable_bins(validation)))
    plot_solid_fraction_distribution(validation, bins=20, include_missing_hist=False)
    plt.title("Validation Data: Solid Fraction Distribution")
    save_plot("validation_solid_fraction_distribution.png")
    print("Saved plot to validation_solid_fraction_distribution.png")

    print("Loading test data...")
    test = load_test_data(suppress_load_errors=True)
    plt.figure()
    print("Number of total test bins: ", len(test))
    print("Number of usable test bins: ", len(get_usable_bins(test)))
    plot_solid_fraction_distribution(test, bins=20, include_missing_hist=False)
    plt.title("Test Data: Solid Fraction Distribution")
    save_plot("test_solid_fraction_distribution.png")
    print("Saved plot to test_solid_fraction_distribution.png")