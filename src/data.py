import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def prepare_train_valid_test(df, valid_prop=.15, test_prop=.15):
    """
    df is a Pandas dataframe
    """
    # The first step is to create train, validation, and test datasets.
    # Assume that we want 15% of the data to be in the test set, 15%
    # to be in the validation set, and the rest to be in the train set.
    train_test_idx = int(len(df) * (1-test_prop))
    train_valid_idx = int((train_test_idx*valid_prop) / (1-test_prop))
    df_0 = df.iloc[:train_test_idx, :]
    df_test = df.iloc[train_test_idx:, :]

    # Min-max scale the times and amounts in df_0 for numerical stability during
    # training evaluation; since min times and amounts are 0, just divide everything
    # by the largest time/amount in the dataset
    max_time = np.max(df_0['Time'])
    max_amount = np.max(df_0['Amount'])
    df_0['Time'] = df_0['Time'].apply(lambda time: time/max_time)
    df_0['Amount'] = df_0['Amount'].apply(lambda amount: amount/max_amount)

    # Use the max time and amount calculated on the training set to standardize
    # the times and amounts in the test set
    df_test['Time'] = df_test['Time'].apply(lambda time: time/max_time)
    df_test['Amount'] = df_test['Amount'].apply(lambda amount: amount/max_amount)

    # Shuffle the train-validation dataframe to reduce bias
    df_0 = df_0.sample(frac=1)

    # Generate train and validation sets
    df_valid = df_0.iloc[:train_valid_idx, :]
    df_train = df_0.iloc[train_valid_idx:, :]

    return df_train, df_valid, df_test


class FraudDataset(Dataset):
    def __init__(self, df):
        """
        Assumes that the target variable is the final column of the passed
        in dataframe
        
        df is a Pandas dataframe
        """
        df_X = df.iloc[:, :-1]
        df_y = df.iloc[:, -1]
        self.X = torch.tensor(df_X.values).float()
        self.y = torch.nn.functional.one_hot(torch.tensor(df_y.values)).float()


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
