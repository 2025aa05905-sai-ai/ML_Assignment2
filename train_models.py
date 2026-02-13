import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

print("Loading dataset...")
df = pd.read_csv("diabetes_data.csv")

target_colmn = "Diabetes_binary"

X = df.drop(target_colmn, axis=1)
y = df[target_colmn]

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


train_df = X_train.copy()
train_df[target_colmn] = y_train
train_df.to_csv("train.csv", index=False)

test_df = X_test.copy()
test_df[target_colmn] = y_test
test_df.to_csv("test.csv", index=False)

print("Training models...")

models = {
    "logistic": LogisticRegression(max_iter=2000),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, pred)
    mcc = matthews_corrcoef(y_test, pred)

    print(f"{name} Accuracy:", acc)
    print(f"{name} Precision:", prec)
    print(f"{name} Recall:", rec)
    print(f"{name} F1:", f1)
    print(f"{name} AUC:", auc)
    print(f"{name} MCC:", mcc)

    joblib.dump(model, f"model/{name}.pkl")

print("\nTrain.csv & Test.csv created")
print("All models saved inside folder")
