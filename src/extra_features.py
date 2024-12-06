class Income_Municipality:
    def __init__(self, df, df_income):
        """
        Initializes the extra feature with a DataFrame.
        """
        self.df = df
        self.df_income = df_income

    def add_feature(self):
        """Merges self.df with self.df_income to add the 'Average_Income_Per_Citizen' feature."""

        self.df = self.df.merge(
            self.df_income[["CD_MUNTY_REFNIS", "Average_Income_Per_Citizen"]],
            left_on="municipality_code",
            right_on="CD_MUNTY_REFNIS",
            how="left",
        )
        print("New fature added to the DataFrame: Average_Income_Per_Citizen")
        return self.df
