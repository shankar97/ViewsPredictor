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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import tree
import graphviz




# pdb.set_trace()
# dataset = numpy.loadtxt("download (14).csv", delimiter=",",skiprows=1)
dataset = pd.read_csv("dataset/download (18).csv")
dataset = dataset.drop("DislikeCount",axis=1)
dataset = dataset.drop("DislikeCountRounded",axis=1)
dataset = dataset.drop("LikeDislikeRatio",axis=1)
# print dataset.describe()

df2 = dataset.columns.get_values()

df2 = df2.tolist()

X = dataset.drop("LikedOrDisliked",axis = 1)
Y = dataset["LikedOrDisliked"]

## MLP implmented using 100 layers
X_train, X_test, y_train, y_test = train_test_split(X, Y)
# pdb.set_trace()
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(100,200,100),max_iter=600)
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)

print(classification_report(y_test,predictions))
print "MLP accuracy using 100 hidden layers: " + str(accuracy_score(y_test,predictions)*100)

## Decision Tree implementation
# pdb.set_trace()
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3)

clf_gini = DecisionTreeClassifier(criterion = "gini", max_depth=10, min_samples_leaf=2)
clf_gini.fit(X_train, y_train)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth=10, min_samples_leaf=2)
clf_entropy.fit(X_train, y_train)

# # regs = DecisionTreeRegressor(max_depth = 5)
# regs.fit(X_train,y_train)

pred_gini = clf_gini.predict(X_test)
pred_entropy = clf_entropy.predict(X_test)
# pred_reg = regs.predict(X_test)

print "Tree Accuracy Using gini index: " + str(accuracy_score(y_test,pred_gini)*100)
print "Tree Accuracy Using info gain: " + str(accuracy_score(y_test,pred_entropy)*100)
# print accuracy_score(y_test,pred_reg)

doten = tree.export_graphviz(clf_entropy,out_file='tree_en.dot',feature_names=df2,filled=True,rounded=True)
dotgini = tree.export_graphviz(clf_gini,out_file='tree_gini.dot',feature_names=df2,filled=True,rounded=True)




