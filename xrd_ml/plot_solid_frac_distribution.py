import matplotlib.pyplot as plt
from argparse import ArgumentParser

from plotting import (
    plot_solid_fraction_distribution, 
    plot_solid_fraction_distribution_from_np_array,
    save_plot,
    set_plots_subdirectory
)
from load_data import (
    load_processed_data,
    get_usable_bins
)
from train_test_split import (
    load_train_data,
    load_validation_data,
    load_test_data,
    get_x_y_as_np_array,
    TRAIN_TEST_SPLITS,
)
from imbalance import resample_dataset_from_binned_solid_fractions


if __name__ == "__main__":

    # Parse command line argument to determine the TRAIN_TEST_SPLIT
    parser = ArgumentParser(description="Plot solid fraction distribution.")
    parser.add_argument(
        "--split",
        type=str,
        default="train_2000_val_2500_test_3500",
        choices=TRAIN_TEST_SPLITS.keys(),
        help="Specify the train-test split to use (keys from train_test_split.py.TRAIN_TEST_SPLIT).",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Whether to include a plot of the balanced train dataset."
    )
    args = parser.parse_args()
    print(f"Using train test split: {args.split}")
    split = TRAIN_TEST_SPLITS[args.split]

    set_plots_subdirectory(f"solid_frac_distribution_{args.split}_split", add_timestamp=True)

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
    train = load_train_data(split=split, suppress_load_errors=True)
    print("Number of total training bins: ", len(train))
    print("Number of usable training bins: ", len(get_usable_bins(train)))
    plt.figure()
    plot_solid_fraction_distribution(train, bins=20, include_missing_hist=False)
    plt.title("Training Data: Solid Fraction Distribution")
    save_plot("train_solid_fraction_distribution.png")
    print("Saved plot to train_solid_fraction_distribution.png")

    print("Loading validation data...")
    validation = load_validation_data(split=split, suppress_load_errors=True)
    plt.figure()
    print("Number of total validation bins: ", len(validation))
    print("Number of usable validation bins: ", len(get_usable_bins(validation)))
    plot_solid_fraction_distribution(validation, bins=20, include_missing_hist=False)
    plt.title("Validation Data: Solid Fraction Distribution")
    save_plot("validation_solid_fraction_distribution.png")
    print("Saved plot to validation_solid_fraction_distribution.png")

    print("Loading test data...")
    test = load_test_data(split=split, suppress_load_errors=True)
    plt.figure()
    print("Number of total test bins: ", len(test))
    print("Number of usable test bins: ", len(get_usable_bins(test)))
    plot_solid_fraction_distribution(test, bins=20, include_missing_hist=False)
    plt.title("Test Data: Solid Fraction Distribution")
    save_plot("test_solid_fraction_distribution.png")
    print("Saved plot to test_solid_fraction_distribution.png")

    if args.balance:
        print("Balancing training data ...")
        train_x, train_y = get_x_y_as_np_array(train)
        balanced_train_x, balanced_train_y = resample_dataset_from_binned_solid_fractions(
            data=train_x,
            solid_fractions=train_y,
            n_bins=20,
            bin_undersampling_threshold=0.8,
            oversample=False,
            random_seed=42
        )
        plt.figure()
        plot_solid_fraction_distribution_from_np_array(balanced_train_y, bins=20)
        plt.title("Balanced Training Data: Solid Fraction Distribution")
        save_plot("balanced_train_solid_fraction_distribution.png")
        print("Saved plot to balanced_train_solid_fraction_distribution.png")