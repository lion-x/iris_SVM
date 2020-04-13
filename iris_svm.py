import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def iris_type(s):
    lables={b'Iris-setosa':0, b'Iris-versicolor':1, b'Iris-virginica':2,}
    return lables[s]

filepath = 'iris.txt'
data = np.loadtxt(filepath, dtype=float, delimiter=',', converters={4:iris_type})
print(data)
X, y = np.split(data, (4,), axis=1)
x_train,  x_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=23)

model = svm.SVC(gamma=0.1, C=0.8, decision_function_shape='ovo')
model.fit(x_train, y_train.ravel())
predict = model.predict(x_test)
print(accuracy_score(predict, y_test))
print(recall_score(predict, y_test, average=None))

x = X[:,0:2]
x_train,  x_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=23)
model = svm.SVC(gamma=0.1, C=0.8, decision_function_shape='ovo')
model.fit(x_train, y_train.ravel())
predict = model.predict(x_test)
plt.figure()
plt.subplot(121)
plt.scatter(x_test[:,0], x_test[:,1], c = y_test.reshape((-1)), s=50)

plt.subplot(122)
plt.scatter(x_test[:,0], x_test[:,1], c = predict.reshape((-1)), s=50)
plt.show()