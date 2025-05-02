from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import accuracy_score, classification_report

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_preds))
    print(classification_report(y_test, y_preds))