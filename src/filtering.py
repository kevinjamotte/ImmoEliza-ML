def postal_filtering(df, x):
    # value of 20 shows best results
    # Filter out postal codes
    postal_code_counts = df['postal_code'].value_counts()
    valid_postal_codes = postal_code_counts[postal_code_counts >= x].index 
    print(f"Original DataFrame shape: {df.shape}")
    # Create a filtered DataFrame
    df = df[df['postal_code'].isin(valid_postal_codes)]

    # Print the result
    print(f"Filtered DataFrame shape: {df.shape}")
    return df