import pandas as pd
import os


class DataProcessor:
    def __init__(self, possible_versions):
        self.possible_versions = possible_versions
        self.data_by_version = {}

    def load_data(self):
        for version in self.possible_versions:
            # Load CSV file
            data = pd.read_csv(os.path.join("Item Data",f"{version}_item_data.csv"))
            self.data_by_version[version] = data

    def preprocess_data(self):
        for version, data in self.data_by_version.items():
            # Calculate average stat per gold for each stat
            stats_columns = [col for col in data.columns if col not in ['item_id', 'name', 'total_gold']]
            for stat_col in stats_columns:
                data[f'avg_{stat_col}_per_gold'] = data[stat_col] / data['total_gold']
            # Compare average stat per gold across different versions (if needed)

    def split_data(self, test_indices, verification_indices):
        # Combine the processed data for all versions into a single dataset
        print(self.data_by_version)
        all_data = pd.concat(self.data_by_version.values(), ignore_index=True)

        # Split the combined dataset into training, testing, and verification sets
        train_data = all_data[~all_data.index.isin(test_indices + verification_indices)]
        test_data = all_data.loc[test_indices]
        verification_data = all_data.loc[verification_indices]

        return train_data, test_data, verification_data
