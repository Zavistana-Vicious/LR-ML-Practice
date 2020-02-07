import numpy as np
from sklearn import datasets

# Load the data
iris = datasets.load_iris()
list(iris.keys())
x = iris["data"][:, :4]
y = (iris["target"] == 2).astype(np.int)

# Train LR Model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x, y)

# Model's estimated probabilities
import matplotlib.pyplot as plt
fig1 = plt.subplot()
x_0 = np.linspace(4.5, 8, 1000)
x_1 = np.linspace(2.5, 4.5, 1000)
x_2 = np.linspace(1, 7, 1000)
x_3 = np.linspace(0, 3, 1000)
x_new = np.concatenate([[x_0], [x_1], [x_2], [x_3]]).transpose()
y_proba = log_reg.predict_proba(x_new)
fig1.plot(x_new[:, 0], y_proba[:, 1], "g-", label="sepal length")
fig1.plot(x_new[:, 1], y_proba[:, 1], "b-", label="sepal width")
fig1.plot(x_new[:, 2], y_proba[:, 1], "r-", label="petal length")
fig1.plot(x_new[:, 3], y_proba[:, 1], "y-", label="petal width")
fig1.legend()
fig1.set_xlabel("cm")
fig1.set_ylabel("probability of virginica")
fig1.set_title("probability of virginica vs features.;")

# compare petal width and length
'''fig2 = plt.subplot()
fig2.plot(x_new[:, 2], x_new[:, 3])'''
