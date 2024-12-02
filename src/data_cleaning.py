def cleaner(df):
    df.dropna() #making sure we have no missing values
    df.drop_duplicates() #dropping the duplicates
    df['municipality_code'] = df['municipality_code'].astype('int') #changing type to int
    df['facades'] = df['facades'].astype('int') #changing type to int
    df.drop(['price_sqm'], axis=1, inplace=True) #dropping price_sqm column as it has high correlation with price
    

 

