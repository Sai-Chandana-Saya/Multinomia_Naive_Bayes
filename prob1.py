import os
import argparse
import pickle
import numpy as np
import pandas as pd
from utils import *
from model import *
import csv


def main(args: argparse.Namespace):
    # Set seed for reproducibility
    set_seed(args.sr_no)

    # Load the data
    X_train, y_train, X_val, y_val = get_data(
        path=os.path.join(args.data_path, args.train_file), seed=args.sr_no)
    print("Data Loaded")

    # Preprocess the data
    vectorizer = Vectorizer(max_vocab_len=args.max_vocab_len)
    vectorizer.fit(X_train)
    if os.path.exists(f"{args.data_path}/X_train{args.intermediate}"):
        X_train_vec = pickle.load(open(
            f"{args.data_path}/X_train{args.intermediate}", "rb"))
        y_train = pickle.load(open(
            f"{args.data_path}/y_train{args.intermediate}", "rb"))
        X_val_vec = pickle.load(open(
            f"{args.data_path}/X_val{args.intermediate}", "rb"))
        y_val = pickle.load(open(
            f"{args.data_path}/y_val{args.intermediate}", "rb"))
        print("Preprocessed Data Loaded")
    else:
        X_train_vec = vectorizer.transform(X=X_train)
        pickle.dump(
            X_train_vec,
            open(f"{args.data_path}/X_train{args.intermediate}", "wb"))
        pickle.dump(
            y_train, open(f"{args.data_path}/y_train{args.intermediate}", "wb"))
        X_val_vec = vectorizer.transform(X=X_val)
        pickle.dump(
            X_val_vec,
            open(f"{args.data_path}/X_val{args.intermediate}", "wb"))
        pickle.dump(
            y_val, open(f"{args.data_path}/y_val{args.intermediate}", "wb"))
        print("Data Preprocessed")

    # Train the model
    model = MultinomialNaiveBayes(alpha=args.smoothing)
    model.fit(X_train_vec, y_train)
    print("Model Trained")

    # Evaluate the trained model
    y_pred_train = model.predict(X_train_vec)
    train_accuracy = np.mean(y_pred_train == y_train)
    print(f"Train Accuracy: {train_accuracy}")

    y_pred_val = model.predict(X_val_vec)
    val_accuracy = np.mean(y_pred_val == y_val)
    print(f"Validation Accuracy: {val_accuracy}")

    # Save validation accuracy to a CSV file
    with open("validation_accuracy.csv", "a", newline="") as f:
        writer = csv.writer(f)
        # Write the header only if the file is empty
        if f.tell() == 0:
            writer.writerow(["SR_No", "Smoothing", "Validation_Accuracy"])
        # Write the validation accuracy with relevant details
        writer.writerow([args.sr_no, args.smoothing, val_accuracy])

    print("Validation accuracy saved in validation_accuracy.csv")

    # Load the test data
    if os.path.exists(f"{args.data_path}/X_test{args.intermediate}"):
        X_test_vec = pickle.load(open(
            f"{args.data_path}/X_test{args.intermediate}", "rb"))
        print("Preprocessed Test Data Loaded")
    else:
        X_test = pd.read_csv(
            f"{args.data_path}/X_test_{args.sr_no}.csv", header=None
        ).values.squeeze()
        print("Test Data Loaded")
        X_test_vec = vectorizer.transform(X=X_test)
        pickle.dump(
            X_test_vec,
            open(f"{args.data_path}/X_test{args.intermediate}", "wb"))
        print("Test Data Preprocessed")

    # Generate predictions for the test data
    preds = model.predict(X_test_vec)
    with open(f"predictions.csv", "w") as f:
        for pred in preds:
            f.write(f"{pred}\n")
    print("Predictions Saved to predictions.csv")
    print("You may upload the file at http://10.192.30.174:8000/submit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True, help="Your 5-digit SR number")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data directory")
    parser.add_argument("--train_file", type=str, default="train.csv", help="Name of the training data file")
    parser.add_argument("--intermediate", type=str, default="_i.pkl", help="Suffix for intermediate files")
    parser.add_argument("--max_vocab_len", type=int, default=10_000, help="Maximum vocabulary size")
    parser.add_argument("--smoothing", type=float, default=0.1, help="Smoothing parameter for Naive Bayes")
    args = parser.parse_args()

    # Ensure the data directory exists
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
        print(f"Created data directory: {args.data_path}")

    main(args)