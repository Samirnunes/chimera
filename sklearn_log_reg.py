import time

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

begin = time.time()

model = SGDClassifier(max_iter=200, epsilon=1e-11, eta0=1e-7)

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

model.fit(X_train, np.array(y_train).ravel())

end = time.time()

print(f"Latency: {end - begin}")
