import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main(args: argparse.Namespace):
    # Check if the logs directory exists
    if not os.path.exists(args.logs_path):
        raise FileNotFoundError(f"Logs directory not found: {args.logs_path}")

    # Initialize lists to store the accuracies for active and random strategies
    active_accuracies = []
    random_accuracies = []

    # Load data for active and random strategies
    for i, strategy in enumerate([True, False]):
        for j in range(1, 6):
            file_path = os.path.join(args.logs_path, f"run_{j}_{strategy}.npy")
            
            # Check if the file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} not found in {args.logs_path}")
            
            # Load the data from the .npy file
            data = np.load(file_path)
            
            # Ensure all files have the same length
            if j == 1:
                reference_length = len(data)
            if len(data) != reference_length:
                raise ValueError(f"File {file_path} has inconsistent length")
            
            # Append the data to the respective list
            if strategy:
                active_accuracies.append(data)
            else:
                random_accuracies.append(data)

    # Convert lists to numpy arrays for easier computation
    active_accuracies = np.array(active_accuracies)
    random_accuracies = np.array(random_accuracies)

    # Compute the mean and standard deviation of the accuracies
    active_mean = np.mean(active_accuracies, axis=0)
    active_std = np.std(active_accuracies, axis=0)
    random_mean = np.mean(random_accuracies, axis=0)
    random_std = np.std(random_accuracies, axis=0)

    # Generate the x-axis (number of labeled samples)
    num_samples = np.arange(10_000, 10_000 + 5_000 * len(active_mean), 5_000)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(num_samples, active_mean, label="Active Strategy", color="blue")
    plt.fill_between(num_samples, active_mean - active_std, active_mean + active_std, color="blue", alpha=0.2)
    plt.plot(num_samples, random_mean, label="Random Strategy", color="red")
    plt.fill_between(num_samples, random_mean - random_std, random_mean + random_std, color="red", alpha=0.2)
    
    # Plot the supervised accuracy as a horizontal line
    plt.axhline(y=args.supervised_accuracy, color="green", linestyle="--", label="Supervised Accuracy")
    
    # Add labels and title
    plt.xlabel("Number of Labeled Samples")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy over Iterations (SR No: {args.sr_no})")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig("active_learning_plot.png")
    print("Plot saved as active_learning_plot.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True, help="Your 5-digit SR number")
    parser.add_argument("--logs_path", type=str, default="logs", help="Path to the logs directory")
    parser.add_argument("--supervised_accuracy", type=float, required=True, help="Supervised baseline accuracy")
    args = parser.parse_args()

    main(args)