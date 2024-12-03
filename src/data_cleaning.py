def cleaner(df):
    df.dropna() #making sure we have no missing values
    df.drop_duplicates() #dropping the duplicates
    df.drop(['price_sqm'], axis=1, inplace=True) #dropping price_sqm column as it has high correlation with price
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df.drop(['kot'], axis=1, inplace=True)
    df.drop(['loft'], axis=1, inplace=True)
    df.drop(['apartment_block'], axis=1, inplace=True)
    df.drop(['ground_floor'], axis=1, inplace=True)
    df.drop(['country_cottage'], axis=1, inplace=True)
    df.drop(['mansion'], axis=1, inplace=True)
    df.drop(['penthouse'], axis=1, inplace=True)
    df.drop(['exceptional_property'], axis=1, inplace=True)
    df.drop(['manor_house'], axis=1, inplace=True)
    df.drop(['service_flat'], axis=1, inplace=True)
    df.drop(['chalet'], axis=1, inplace=True)
    df.drop(['locality'], axis=1, inplace=True)
    df.drop(['province'], axis=1, inplace=True)
    df.drop(['CD_MUNTY_REFNIS'], axis=1, inplace=True)
    return df

    




