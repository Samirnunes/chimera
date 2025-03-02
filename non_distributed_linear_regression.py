import pandas as pd
from sklearn.linear_model import SGDRegressor

r = SGDRegressor(max_iter=100, eta0=1e-7)

X_train = pd.read_csv("./chimera_train_data/sgd1/X_train.csv").drop(
    ["index"], axis=1
)
y_train = pd.read_csv("./chimera_train_data/sgd1/y_train.csv")

r.fit(X_train, y_train)

X_pred_columns = [
    "gre_score",
    "toefl_score",
    "university_rating",
    "sop",
    "lor",
    "cgpa",
    "research",
]
X_pred_rows = [
    [337, 118, 4, 4.5, 4.5, 9.65, 1],
    [324, 107, 4, 4, 4.5, 8.87, 1],
    [314, 103, 2, 2, 3, 8.21, 0],
    [333, 117, 4, 5, 4, 9.66, 1],
]

X_pred = pd.DataFrame(X_pred_rows, columns=X_pred_columns)
predictions = r.predict(X_pred)

print(predictions)
