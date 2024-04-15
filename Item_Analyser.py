#main file to train Algorithm
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from DataProcessor import DataProcessor

possible_versions = ["14.1.1", "14.2.1", "14.3.1", "14.4.1", "14.5.1", "14.6.1", "14.7.1"]
test_version = "14.1.1"
verification_version = "14.7.1"
test_indices = [0, 1, 2]  # Example test indices
verification_indices = [100, 101, 102]  # Example verification indices

# Initialize DataProcessor object
processor = DataProcessor(possible_versions)

# Load data
processor.load_data()

# Preprocess data
processor.preprocess_data()

# Split data into training, testing, and verification sets
train_data, test_data, verification_data = processor.split_data(test_indices, verification_indices)

# Now you have the train, test, and verification data ready for further processing

with open("test.csv",'w') as test:
    test.write(test_data.to_csv())
with open("train.csv",'w') as test:
    test.write(train_data.to_csv())
with open("verification.csv",'w') as test:
    test.write(verification_data.to_csv())


# Print the shape of each dataset
print("Shape of train data:", train_data.shape)
print("Shape of test data:", test_data.shape)
print("Shape of verification data:", verification_data.shape)

# Print the first few rows of each dataset
print("\nFirst few rows of train data:")
print(train_data.head())

print("\nFirst few rows of test data:")
print(test_data.head())

print("\nFirst few rows of verification data:")
print(verification_data.head())

# You can also access other attributes of the dataframes, such as columns, dtypes, etc.
# For example:
print("\nColumns of train data:")
print(train_data.columns)

# Accessing specific attributes
# For example:
print("\nTotal gold of the first item in train data:", train_data.iloc[0]['total_gold'])

