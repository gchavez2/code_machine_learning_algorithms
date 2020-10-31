"""
- supervised learning is also known as function approximation because ultimately what you are doing is finding a function that matches your training examples well
- you start with some general form of the function (e.g. y = mx+b) and then you tune the parameters such that it best describes your training examples (i.e. change m and b until you get a line that best splits your data)

 Supervised learning is just function approximation. You start with a general function and then tweak the parameters of the function based on your training examples until your function describes the training data well.
"""

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.5)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier()

# Train and inference
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))