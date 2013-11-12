import numpy as np

# Create a the dataset
X = np.linspace(0, 6, 100)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Fit regression model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

#clf_1 = DecisionTreeRegressor(max_depth=4)
clf_1 = RandomForestRegressor(n_estimators=300, n_jobs=1, verbose=1)
clf_2 = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=300)

clf_1.fit(X, y)
clf_2.fit(X, y)

# Predict
y_1 = clf_1.predict(X)
y_2 = clf_2.predict(X)

# Plot the results
import pylab as pl

pl.figure()
pl.scatter(X, y, c="k", label="training samples")
pl.plot(X, y_1, c="g", label="Random Forest", linewidth=2)
pl.plot(X, y_2, c="r", label="AdaBoost", linewidth=2)
pl.xlabel("data")
pl.ylabel("target")
pl.title("Boosted Decision Tree Regression")
pl.legend()
pl.show()