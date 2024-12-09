import pandas as pd


class One_Hot:
    def __init__(self, df, column_name):
        """
        Initializes the One Hot encoder
        """
        self.df = df
        self.column_name = column_name

    def one_hot_encoder(self):
        """
        Creates different columns for each value.
        """
        self.df = pd.get_dummies(self.df, columns=[self.column_name])
        return self.df


class Postal_Filtering:
    def __init__(self, df, x):
        """
        Initializes the filter.
        """
        self.df = df
        self.x = x

    def postal_filtering(self):
        """
        Filters postcode with more than X properties.
        """
        postal_code_counts = self.df["postal_code"].value_counts()
        valid_postal_codes = postal_code_counts[postal_code_counts >= self.x].index

        print(f"DataFrame before filtering: {self.df.shape}")

        self.df = self.df[self.df["postal_code"].isin(valid_postal_codes)]

        print(f"DataFrame after filtering: {self.df.shape}")

        return self.df


class BedroomsFiltering:
    def __init__(self, df, x):
        """
        Initializes the filter.
        """
        self.df = df
        self.x = x

    def bedrooms_filtering(self):
        self.df = self.df[self.df["bedrooms"] < self.x]

        return self.df
