class DataFrameCleaner:
    def __init__(self, df):
        """
        Initializes the cleaner with a DataFrame.
        """
        self.df = df
        print(f"DataFrame before cleaning: {self.df.shape}")

    def drop_na(self):
        """Drops rows with NaN values in specific columns."""
        self.df = self.df.dropna(subset=["Average_Income_Per_Citizen"])
        return self

    def drop_duplicates(self):
        """Drops duplicate rows."""
        self.df = self.df.drop_duplicates()
        return self

    def drop_columns(self):
        """Drops unnecessary columns, 1 line per column allows easy visualisation and feature tuning"""
        self.df.drop(["price_sqm"], axis=1, inplace=True)
        self.df.drop(["Unnamed: 0"], axis=1, inplace=True)
        self.df.drop(["Unnamed: 0.1"], axis=1, inplace=True)
        self.df.drop(["Unnamed: 0.2"], axis=1, inplace=True)
        self.df.drop(["Unnamed: 0.3"], axis=1, inplace=True)
        self.df.drop(["Unnamed: 0.4"], axis=1, inplace=True)
        self.df.drop(["kot"], axis=1, inplace=True)
        self.df.drop(["loft"], axis=1, inplace=True)
        self.df.drop(["apartment_block"], axis=1, inplace=True)
        self.df.drop(["ground_floor"], axis=1, inplace=True)
        self.df.drop(["country_cottage"], axis=1, inplace=True)
        self.df.drop(["mansion"], axis=1, inplace=True)
        self.df.drop(["penthouse"], axis=1, inplace=True)
        self.df.drop(["exceptional_property"], axis=1, inplace=True)
        self.df.drop(["manor_house"], axis=1, inplace=True)
        self.df.drop(["service_flat"], axis=1, inplace=True)
        self.df.drop(["chalet"], axis=1, inplace=True)
        self.df.drop(["locality"], axis=1, inplace=True)
        # self.df.drop(['province'], axis=1, inplace=True)
        self.df.drop(["CD_MUNTY_REFNIS"], axis=1, inplace=True)
        self.df.drop(["pool"], axis=1, inplace=True)
        self.df.drop(["municipality_code"], axis=1, inplace=True)
        self.df.drop(["fireplace"], axis=1, inplace=True)
        self.df.drop(["furnished"], axis=1, inplace=True)
        self.df.drop(["postal_code"], axis=1, inplace=True)
        self.df.drop(["terrace"], axis=1, inplace=True)
        self.df.drop(["garden"], axis=1, inplace=True)
        # self.df.drop(['to_restore'], axis=1, inplace=True)
        # self.df.drop(['to_renovate'], axis=1, inplace=True)
        # self.df.drop(['to_be_done_up'], axis=1, inplace=True)
        # self.df.drop(['kitchen'], axis=1, inplace=True)
        # self.df.drop(['is_house'], axis=1, inplace=True)
        # self.df.drop(['is_apartment'], axis=1, inplace=True)
        # self.df.drop(['just_renovated'], axis=1, inplace=True)
        # self.df.drop(['good'], axis=1, inplace=True)
        # self.df.drop(['facades'], axis=1, inplace=True)
        print(f"DataFrame after cleaning: {self.df.shape}")

        return self

    def clean(self):
        """Executes the entire cleaning process."""
        return self.drop_na().drop_duplicates().drop_columns().df
