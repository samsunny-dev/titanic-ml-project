from src.data_cleaning import load_data, clean_data
from src.utils import split_data
from src.model import train_model, evaluate_model

def main():
    df = load_data("data/titanic.csv")
    df_clean = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df_clean)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
