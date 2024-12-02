def cleaner(df):
    df.dropna() #making sure we have no missing values
    df.drop_duplicates() #dropping the duplicates
    df['facades'] = df['facades'].astype('int') #changing type to int
    df.drop(['price_sqm'], axis=1, inplace=True) #dropping price_sqm column as it has high correlation with price
    df.drop(['municipality_code'], axis=1, inplace=True) #dropping municipality code column as postal codes showed better accuracy
    df = pd.get_dummies(df, columns=['postal_code'], drop_first=False)
   


 

