def income_municipality_func(df, income_municipality):
    df = df.merge(
    income_municipality[['CD_MUNTY_REFNIS', 'Average_Income_Per_Citizen']],
    left_on='municipality_code',
    right_on='CD_MUNTY_REFNIS',
    how='left'
    )
    print(df)
    return df