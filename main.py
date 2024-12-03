import pandas as pd  # For data manipulation and DataFrame handling
import numpy as np  # For numerical operations and handling missing values
# Importing necessary libraries from scikit-learn
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # For feature scaling
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.tree import plot_tree  # For visualizing the decision tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Importing libraries for plotting
import matplotlib.pyplot as plt  # For basic plotting
import seaborn as sns  # For advanced data visualization

# Importing modules
from src.data_cleaning import cleaner
from src.extra_features import income_municipality_func
from src.filtering import postal_filtering
from src.outliers import zscore_method
from src.test_train import test_train_func


dataset = './data/raw/dataset_province_municipality_code.csv'
income_municipality = './data/raw/income_municipality.csv'
df = pd.read_csv(dataset)
df_income = pd.read_csv(income_municipality)

df =income_municipality_func(df, df_income)

df = cleaner(df)

df = postal_filtering(df, 20)

df = zscore_method(df, 3)
print(df.dtypes)
df = pd.get_dummies(df, columns=['postal_code'], drop_first=False)
X_train, X_test, y_train, y_test, df_train, df_test = test_train_func(df)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform on the training set
X_test = scaler.transform(X_test)       # Only transform the test set

# Initialize the RandomForestRegressor model
model = RandomForestRegressor(random_state=42, n_estimators=100)  # You can adjust n_estimators as needed

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
