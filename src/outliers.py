from scipy.stats import zscore

def quantiles_method(df):
    Q1 = df[['bedrooms', 'price', 'livingarea', 'surfaceoftheplot', 'gardensurface']].quantile(0.15)
    Q3 = df[['bedrooms', 'price', 'livingarea', 'surfaceoftheplot', 'gardensurface']].quantile(0.85)
    IQR = Q3 - Q1

    # Define the outlier condition: values outside 1.5 * IQR
    outliers_condition = ((df[['bedrooms', 'price', 'livingarea', 'surfaceoftheplot', 'gardensurface']] < (Q1 - 1.5 * IQR)) | 
                        (df[['bedrooms', 'price', 'livingarea', 'surfaceoftheplot', 'gardensurface']] > (Q3 + 1.5 * IQR)))

    # Remove rows where any of the relevant columns contain outliers
    df = df[~outliers_condition.any(axis=1)]
    return df

def zscore_method(df):
    from scipy.stats import zscore
    z_scores = df[['bedrooms', 'price', 'livingarea', 'surfaceoftheplot', 'gardensurface', 'facades']].apply(zscore)
    threshold = 3
    df = df[(z_scores < threshold).all(axis=1)]
    return df



