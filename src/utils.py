from sklearn.model_selection import train_test_split

def split_data(titanic_df, target_column="Survived"):
    X = titanic_df.drop(target_column, axis=1)
    y = titanic_df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)