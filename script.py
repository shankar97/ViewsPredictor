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
from sklearn.model_selection import cross_val_score
from sklearn.tree._tree import TREE_LEAF

def prune_index(inner_tree, index, threshold):
    if inner_tree.value[index].min() < threshold:
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are shildren, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        prune_index(inner_tree, inner_tree.children_left[index], threshold)
        prune_index(inner_tree, inner_tree.children_right[index], threshold)


# pdb.set_trace()
# dataset = numpy.loadtxt("download (14).csv", delimiter=",",skiprows=1)
dataset = pd.read_csv("dataset/download (18).csv")
# dataset = dataset.drop("DislikeCount",axis=1)
# dataset = dataset.drop("DislikeCountRounded",axis=1)
# dataset = dataset.drop("LikeDislikeRatio",axis=1)
dataset = dataset.drop("LikeCount",axis=1)
dataset = dataset.drop("ViewCount",axis=1)
# print dataset.describe()

df2 = dataset.columns.get_values()

df2 = df2.tolist()

X = dataset.drop("LikedOrDisliked",axis = 1)
Y = dataset["LikedOrDisliked"]

acc_MLP = []
acc_DTG = []
acc_DTE = []

## MLP implmented using 100 layers
X_train, X_test, y_train, y_test = train_test_split(X, Y)
# pdb.set_trace()
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(100,200,100),max_iter=600)
mlp.fit(X_train,y_train)
for i in range(0,100):
	mlp.fit(X_train,y_train)
	predictions = mlp.predict(X_test)
	acc_MLP.append(accuracy_score(y_test,predictions)*100)

scores_MLP = cross_val_score(mlp, X, Y, cv=10)

print "MLP accuracy using 100 hidden layers: " + str(sum(acc_MLP)/len(acc_MLP))

## Decision Tree implementation
# pdb.set_trace()
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3)

clf_gini = DecisionTreeClassifier(criterion = "gini", max_depth=10, min_samples_leaf=5)
clf_gini.fit(X, Y)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", max_depth=10, min_samples_leaf=5)
clf_entropy.fit(X, Y)

# # regs = DecisionTreeRegressor(max_depth = 5)
# regs.fit(X_train,y_train)

for i in range(0,100):
	pred_gini = clf_gini.predict(X_test)
	pred_entropy = clf_entropy.predict(X_test)
	# pred_reg = regs.predict(X_test)
	acc_DTE.append(accuracy_score(y_test,pred_entropy)*100)
	acc_DTG.append(accuracy_score(y_test,pred_gini)*100)

# scores_DTE = cross_val_score(clf_entropy, X, Y, cv=300)
# scores_DTG = cross_val_score(clf_gini, X, Y, cv=300)
# pdb.set_trace()
print "Tree Accuracy Using gini index: " + str(sum(acc_DTG)/len(acc_DTG))
print "Tree Accuracy Using info gain: " + str(sum(acc_DTE)/len(acc_DTE))
# print accuracy_score(y_test,pred_reg)

doten = tree.export_graphviz(clf_entropy,out_file='tree_en.dot',feature_names=df2,filled=True,rounded=True)
dotgini = tree.export_graphviz(clf_gini,out_file='tree_gini.dot',feature_names=df2,filled=True,rounded=True)




