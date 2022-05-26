# Jay Mistry
# 23859979
# Project 1

import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter

#https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,precision_recall_fscore_support, confusion_matrix

class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None
    

class DecisionTreeModel:

    def __init__(self, max_depth=100, criterion = 'gini', min_samples_split=2, impurity_stopping_threshold = 0):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # TODO  
        X = np.array(X) # Change the Dataframe into Multidimensional Array
        y = np.array(y) # Change the numpy arrays in the Series label
        self._fit(X, y) # Call the _fit method
        # end TODO

    def predict(self, X: pd.DataFrame):
        # TODO
        X = np.array(X) # Numpy nd array is converted into Dataframe
        return self._predict(X) # Call the preditc method
        # end TODO
        
    def _fit(self, X, y):
        self.root = self._build_tree(X, y) #Build Decision Tree Class on X and y.
        
    def _predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X] # repeats every test of datapoint.
        return np.array(predictions) # returns array predictions

    def _is_finished(self, depth):
        if (depth >= self.max_depth # If the max depth and min split is not reached, the condition is False, otherwise the condition is True.
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        return False
    
    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape # Adding n_features to the Dataframe's columns and n samples to the rows
        self.n_class_labels = len(np.unique(y)) # Getting the length of unique labels

        # stopping criteria
        if self._is_finished(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)

        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # Bringing the children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)
    

    def _gini(self, y):
        # TODO
        gini = 0
        # If y = type series, it returns True, otherwise it returns False.
        if isinstance(y, pd.Series): 
            p = y.value_counts() / y.shape[0] # The proportion of each label returned by an array
            gini = 1 - np.sum(p**2) # Inserting the gini formula
        else:
            y = pd.Series(y)
            p = y.value_counts() / y.shape[0] # The proportion of each label returned by an array
            gini = 1 - np.sum(p**2) # gini formula
        #end TODO
        return gini
    
    def _entropy(self, y):
        # TODO: 
        y = np.unique(y, return_inverse = True)[1] # If y is categorical varaible, then it coverts into integer
        proportions = np.bincount(y) / len(y) # The proportion of each label returned by an array
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0]) # Entropy formula
        # end TODO
        return entropy
        
    def _create_split(self, X, thresh):
        left_idx = np.argwhere(X <= thresh).flatten() # Adds the left limit of the split
        right_idx = np.argwhere(X > thresh).flatten() # Adds right limit of the split
        return left_idx, right_idx

    def _information_gain(self, X, y, thresh):
        # TODO: 
        # Fix code so it can switch between two standard: gini and entropy 
        if self.criterion=='gini':
            parent_loss = self._gini(y)
            left_idx, right_idx = self._create_split(X, thresh)
            n, n_left, n_right = len(y), len(left_idx), len(right_idx)

            if n_left == 0 or n_right == 0: 
                    return 0
        
            child_loss = (n_left / n) * self._gini(y[left_idx]) + (n_right / n) * self._gini(y[right_idx])
        # end TODO
            return parent_loss - child_loss

        else:
            parent_loss = self._entropy(y)
            left_idx, right_idx = self._create_split(X, thresh)
            n, n_left, n_right = len(y), len(left_idx), len(right_idx)

            if n_left == 0 or n_right == 0: 
              return 0
        
            child_loss = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
        # end TODO
            return parent_loss - child_loss
       
    def _best_split(self, X, y, features):
        '''TODO: 
        This method or shuffling randomly objects is based on 
        the metric of the maximum number of data values,
        for each function of the filter/sort, 
        the previous function of the filter or sort.
        The feature of continuous values between repeated processing. 
        The exponent or function of the threshold pair that 
        matches the lowest mean of the Gini coefficient is an impurity
        because the parent node of the Gini coefficient is an impurity
        is returned asbest_feat_id and best_threshold.
        '''
        split = {'score':- 1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']
    
    def _traverse_tree(self, x, node):
        '''TODO: 
        The traversre tree function put the tree becomes a set of rules
        '''
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

def _frequent_label(y):

        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

class RandomForestModel(object):

    def __init__(self,n_estimators = 5, min_samples_split = 2, max_depth = 3):
        # TODO:
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        # Will add on the  individually trained decision trees
        self.trees = []
        # end TODO
    

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # TODO:
        '''
        Train random forest crowns x and y.
        Input parametrs :X Dataframe or array
        Output lables: Y Series or array
        Function Transform the input type array model trainin
        '''
        X_array = X.to_numpy() # dataframe to arrays
        y_array = y.to_numpy() # series to arrays
        self.trees = []
        # Estimate of repeatability
        for _ in range(self.n_estimators):
            #Decision tree intializing 
            tree = DecisionTreeModel(max_depth = self.max_depth)
            n_sample, n_feat = X.shape # sample = len of input array X
            #get random choices
            bins = np.random.choice(n_sample, n_sample, replace = True) # get random choices in range len of n_samples
            #trained the model on given data
            tree.fit(X_array[bins], y_array[bins])
            #Added to the list of trees
            self.trees.append(tree)
        
        # end TODO
    

    def predict(self, X: pd.DataFrame):
        # TODO:
        '''
        There may be new class label data for predictions.
        :param X: np.Predict a new instance of an array
I       Then return: 
        '''
        #explain: predicitng on the input data
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        #Return to the most common pred
        y_pred = [_frequent_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
        # end TODO
   

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def classification_report(y_test, y_pred):
    # calculation of  precision, recall, f1-score
    # TODO:
    #confusion matrix  called
    matrix = confusion_matrix(y_test, y_pred) 
     #precion formuula
    precision = matrix[0,0] / (matrix[0,0] + matrix[0,1]) 
    # recall the matix
    recall = matrix[0,0] / (matrix[0,0] + matrix[1,0])
    # F1 score calculated by using precision and recall 
    F1 = 2 * (precision * recall) / (precision + recall) 
    D = [[precision, recall, F1]]

  
    result = [precision, recall, F1]
    return (result)
    # end TODO


def confusion_matrix(y_test,y_pred):
    FP = 0 # False Positive
    FN = 0 # False Negative
    TP = 0 # True Positive
    TN = 0 # True Negative
     
    # actual vs pred val zipped
    for actual_value, predicted_value in zip(y_test, y_pred): 
    # let's first see if it's a true (T) or false prediction (F)
        if predicted_value == actual_value: # T?
            if predicted_value == 1: # predicted TP
                TP += 1
            else: # then predicted TN
                TN += 1
        else: # else predicted F?
            if predicted_value == 1: # then predicted FP
                FP += 1
            else: # FN
                FN += 1
            
    our_confusion_matrix = [[TN, FP],[FN, TP]]
    result = np.array(our_confusion_matrix)
    return result
# Converison of the  numpy array to be printed properly as a matrix


def _test():
    
    df = pd.read_csv('breast_cancer.csv')
    
    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    clf = DecisionTreeModel(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)

if __name__ == "__main__":
    _test()
