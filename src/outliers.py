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
