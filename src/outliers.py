from scipy.stats import zscore
def zscore_method(df, x):
    #3 gives the best results
    print(f"Before ZSCORE DataFrame shape: {df.shape}")
    z_scores = df[['price', 'livingarea']].apply(zscore)
    threshold = x 
    df = df[(z_scores < threshold).all(axis=1)]
    print(f"ZSCORED DataFrame shape: {df.shape}")
    return df



