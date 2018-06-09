import numpy as np
# import scikit
# from keras.models import Sequential
# from keras.layers import Dense
import pdb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree



## MLP implmented using 1000 layers
# pdb.set_trace()
# dataset = numpy.loadtxt("download (14).csv", delimiter=",",skiprows=1)
dataset = pd.read_csv("dataset/download (14).csv")
# print dataset.describe()

X = dataset.drop("LikedOrDisliked",axis = 1)
Y = dataset["LikedOrDisliked"]

# X_train, X_test, y_train, y_test = train_test_split(X, Y)
# # pdb.set_trace()
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# mlp = MLPClassifier(hidden_layer_sizes=(1000,1000,1000),max_iter=500)
# mlp.fit(X_train,y_train)

# predictions = mlp.predict(X_test)

# print(classification_report(y_test,predictions))
# print accuracy_score(y_test,predictions)

## Decision Tree implementation
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3)

clf_gini = DecisionTreeClassifier(criterion = "gini", max_depth=10, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth=10, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

pred_gini = clf_gini.predict(X_test)
pred_entropy = clf_entropy.predict(X_test)

print accuracy_score(y_test,pred_gini)
print accuracy_score(y_test,pred_entropy)

tree.export_graphviz(clf_entropy,out_file='tree_en.dot')
tree.export_graphviz(clf_gini,out_file='tree_gini.dot')

# X = dataset[:,0:20]
# Y = dataset[:,20]

# pdb.set_trace()
# #Creating model for network usig Keras
# model = Sequential()
# model.add(Dense(12, input_dim=7, activation='relu'))
# # model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# model.fit(X, Y, epochs=10, batch_size=10)
# pdb.set_trace()
# scores = model.evaluate(X, Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# def train():
# 	pass

# def test():
# 	pass

