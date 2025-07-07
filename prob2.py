import os
import argparse
import pickle
import numpy as np
from utils import *
from model import *


def main(args: argparse.Namespace):
    # Set seed for reproducibility
    assert args.run_id is not None and 0 < args.run_id < 6, "Invalid run_id"
    set_seed(args.sr_no + args.run_id)

    # Load the preprocessed data
    if os.path.exists(f"{args.data_path}/X_train{args.intermediate}"):
        X_train_vec = pickle.load(open(
            f"{args.data_path}/X_train{args.intermediate}", "rb"))
        X_val_vec = pickle.load(open(
            f"{args.data_path}/X_val{args.intermediate}", "rb"))
        y_train = pickle.load(open(
            f"{args.data_path}/y_train{args.intermediate}", "rb"))
        y_val = pickle.load(open(
            f"{args.data_path}/y_val{args.intermediate}", "rb"))
        idxs = np.random.RandomState(args.run_id).permutation(X_train_vec.shape[0])
        X_train_vec = X_train_vec[idxs]
        y_train = y_train[idxs]
        print("Preprocessed Data Loaded")
    else:
        raise Exception("Preprocessed Data not found")

    # Train the model
    model = MultinomialNaiveBayes(alpha=args.smoothing)
    accs = []
    total_items = 10_000
    idxs = np.arange(10_000)
    remaining_idxs = np.setdiff1d(np.arange(X_train_vec.shape[0]), idxs)

    # Train the model incrementally
    for i in range(1, 60):
        X_train_batch = X_train_vec[idxs]
        y_train_batch = y_train[idxs]

        # Fit the model (update if not the first iteration)
        if i == 1:
            model.fit(X_train_batch, y_train_batch)
        else:
            model.fit(X_train_batch, y_train_batch, update=True)

        # Evaluate the model on the validation set
        y_preds = model.predict(X_val_vec)
        val_acc = np.mean(y_preds == y_val)
        print(f"{total_items} items - Val acc: {val_acc}")
        accs.append(val_acc)

        # Select new data points for the next iteration
        if args.is_active:
            # Active Learning Strategy: Select the most uncertain points
            probs = model.predict_proba(X_train_vec[remaining_idxs])
            uncertainties = -np.sum(probs * np.log(probs + 1e-10), axis=1)  # Entropy
            selected_idxs = remaining_idxs[np.argsort(uncertainties)[-5_000:]]
        else:
            # Random selection strategy
            selected_idxs = remaining_idxs[:5_000]

        # Update the indices for the next iteration
        idxs = np.concatenate([idxs, selected_idxs])
        remaining_idxs = np.setdiff1d(remaining_idxs, selected_idxs)
        total_items += 5_000

    # Save the validation accuracies for this run
    accs = np.array(accs)
    np.save(f"{args.logs_path}/run_{args.run_id}_{args.is_active}.npy", accs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True, help="Your 5-digit SR number")
    parser.add_argument("--run_id", type=int, required=True, help="Run ID (1 to 5)")
    parser.add_argument("--is_active", action="store_true", help="Use Active Learning Strategy")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data directory")
    parser.add_argument("--logs_path", type=str, default="logs", help="Path to save logs")
    parser.add_argument("--intermediate", type=str, default="_i.pkl", help="Suffix for intermediate files")
    parser.add_argument("--max_vocab_len", type=int, default=10_000, help="Maximum vocabulary size")
    parser.add_argument("--smoothing", type=float, default=0.1, help="Smoothing parameter for Naive Bayes")
    args = parser.parse_args()

    # Ensure the logs directory exists
    if not os.path.exists(args.logs_path):
        os.makedirs(args.logs_path)
        print(f"Created logs directory: {args.logs_path}")

    main(args)