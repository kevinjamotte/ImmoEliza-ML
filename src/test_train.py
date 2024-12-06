from sklearn.model_selection import train_test_split


def test_train_func(df):
    X = df.drop("price", axis=1)  # Features (all columns except 'price')
    y = df["price"]  # Target variable (price)

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Combine features and target back into DataFrames for X_train and X_test
    df_train = X_train.copy()
    df_train["price"] = y_train

    df_test = X_test.copy()
    df_test["price"] = y_test

    # Check the shape of the splits
    print(
        f"Training set size: {df_train.shape[0]} samples and {df_train.shape[1]} columns"
    )
    print(f"Test set size: {df_test.shape[0]} samples and {df_train.shape[1]} columns")

    return X_train, X_test, y_train, y_test, df_train, df_test
