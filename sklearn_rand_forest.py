import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

begin = time.time()

model = RandomForestClassifier(
    n_estimators=4, max_depth=5, max_leaf_nodes=15, n_jobs=1
)

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

model.fit(X_train, np.array(y_train).ravel())

end = time.time()

print(f"Latency: {end - begin}")
