import pandas as pd

def load_data(path):
    return pd.read_csv(path)


def clean_data(titanic_df):
    titanic_df = titanic_df.drop(["Cabin", "Ticket", "Name", "PassengerId"], axis=1)
    titanic_df["Age"].fillna(titanic_df["Age"].median(), inplace=True)
    titanic_df["Embarked"].fillna(titanic_df["Embarked"].mode()[0], inplace=True)
    titanic_df = pd.get_dummies(titanic_df, drop_first=True)
    return titanic_df

