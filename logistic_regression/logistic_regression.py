import numpy as np
from sklearn import linear_model

# Load data
data = np.loadtxt('./heights_weights.csv', delimiter=',', skiprows=1)
X = data[:,0:2]
y = data[:,2]

# Fit (train) the Logistic Regression classifier
clf = linear_model.LogisticRegression(C=1e40, solver='newton-cg')
fitted_model = clf.fit(X, y)

# Predict
prediction_result = clf.predict([(70,180)])