def get_embed_from_df(df):
    e_cols = [col for col in df.keys() if col[0] == "e" and int(col[1:]) >= 0]
    return df[e_cols].values
