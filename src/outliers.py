from scipy.stats import zscore


class ZScoreFilter:
    def __init__(self, df, columns, threshold=3):
        """
        Initializes the ZScoreFilter class.

        Parameters:
        - df (pd.DataFrame): The DataFrame to filter.
        - columns (list): List of columns to compute z-scores.
        - threshold (float): Z-score threshold for filtering (default: 3).
        """
        self.df = df
        self.columns = columns
        self.threshold = threshold

    def filter(self):
        """
        Filters the DataFrame rows based on z-scores.

        Returns:
        - pd.DataFrame: Filtered DataFrame.
        """
        print(f"DataFrame before ZSCORE: {self.df.shape}")

        # Compute z-scores only for the specified columns
        z_scores = self.df[self.columns].apply(zscore)

        # Filter rows based on the threshold
        self.df = self.df[(z_scores < self.threshold).all(axis=1)]

        print(f"DataFrame aftr ZSCORE: {self.df.shape}")
        return self.df

    def update_threshold(self, new_threshold):
        """
        Updates the z-score threshold.

        Parameters:
        - new_threshold (float): The new threshold value.
        """
        print(f"Updating threshold from {self.threshold} to {new_threshold}.")
        self.threshold = new_threshold


class OutlierRemover:
    def __init__(self, df, columns, threshold=1.5):
        """
        Initializes the OutlierRemover with a DataFrame and IQR multiplier threshold.

        Parameters:
            df (pd.DataFrame): The DataFrame to process.
            columns (list): List of columns to check for outliers.
            threshold (float): The IQR multiplier for defining outliers (default is 1.5).
        """
        self.df = df
        self.columns = columns
        self.threshold = threshold
        print(f"Initial DataFrame shape: {self.df.shape}")

    def remove_outliers(self):
        """
        Removes outliers from the DataFrame using the IQR method for only the specified columns.

        Outliers are defined as data points lying outside [Q1 - threshold * IQR, Q3 + threshold * IQR].
        """
        for col in self.columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.9)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR

            # Remove rows outside the bounds for this column
            self.df = self.df[
                (self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)
            ]
            print(
                f"Processed column: {col}, lower={lower_bound}, upper={upper_bound}, New shape: {self.df.shape}"
            )

        return self.df
