import pandas as pd
from pandas import DataFrame

class DataFrameCleaner:
    """
    A utility class for cleaning pandas DataFrames. This includes removing duplicates, 
    dropping unnecessary columns, and handling NaN values.
    """
    
    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initializes the cleaner with a DataFrame.

        :param df: A pandas DataFrame to be cleaned.
        """
        self.df: pd.DataFrame = df
        print(f"DataFrame before cleaning: {self.df.shape}")

    def drop_duplicates(self) -> pd.DataFrame:
        """
        Drops duplicate rows in the DataFrame.

        :return: The DataFrameCleaner instance with duplicates removed.
        """
        self.df = self.df.drop_duplicates()
        return self

    def drop_columns(self) -> pd.DataFrame:
        """
        Drops predefined unnecessary columns from the DataFrame.
        Each column is dropped in a single line for easier maintenance.

        :return: The DataFrameCleaner instance with specified columns removed.
        """
        columns_to_drop = [
            "price_sqm", "Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.2", "Unnamed: 0.3",
            "Unnamed: 0.4", "kot", "loft", "apartment_block", "ground_floor",
            "country_cottage", "mansion", "penthouse", "exceptional_property",
            "manor_house", "service_flat", "chalet", "locality", "CD_MUNTY_REFNIS",
            "pool", "municipality_code", "fireplace", "furnished", "postal_code", 
            "garden"
        ]
        
        # Uncomment lines below to optionally drop additional columns
        # "province", "terrace", "to_restore", "to_renovate", "to_be_done_up",
        # "kitchen", "is_house", "is_apartment", "just_renovated", "good", "facades"

        self.df.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')
        return self

    def drop_na(self) -> pd.DataFrame:
        """
        Drops rows with NaN values in the DataFrame.

        :return: The DataFrameCleaner instance with NaN rows removed.
        """
        self.df = self.df.dropna()
        print(f"DataFrame after cleaning: {self.df.shape}")
        return self

    def clean(self) -> pd.DataFrame:
        """
        Executes the complete cleaning process, which includes:
        - Removing duplicate rows.
        - Dropping unnecessary columns.
        - Dropping rows with NaN values.

        :return: The cleaned pandas DataFrame.
        """
        return self.drop_duplicates().drop_columns().drop_na().df
