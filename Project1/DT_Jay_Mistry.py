
import pandas as pd
import numpy as np
import seaborn as sns
import math
import random 
#https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb

from sklearn import datasets

from sklearn.model_selection import train_test_split

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

    def __init__(self, max_depth=100, criterion = 'gini', min_samples_split=2, impurity_stopping_threshold = 1):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # TODO
        # call the _fit method
        self.one_dim = len(np.shape(y)) == 1
        self._fit(X, y)
        self.loss=None
        # end TODO
        print("Done fitting")

    def predict(self, X: pd.DataFrame):
        # TODO
        # call the predict method
        y_pred = self._predict(X)
        # return ...
        return y_pred
        # end TODO
        
    def _fit(self, X, y):
        self.root = self._build_tree(X, y)
        
    def _predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)    
        
    def _is_finished(self, depth):
        # TODO: for graduate students only, add another stopping criteria
        # modify the signature of the method if needed
        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        # end TODO
        return False
    
    def _is_homogenous_enough(self):
        # TODO: for graduate students only
        result = False
        # end TODO
        return result
                              
    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self._is_finished(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)

        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)
    

    def _gini(self, y):
        #TODO
        gini = 0
        proportions = np.bincount(y) / len(y)
        gini = 1 - np.sum(p**2 for p in proportions if p > 0)
        #end TODO
        return gini
    
    def _entropy(self, y):
        # TODO: the following won't work if y is not integer
        # make it work for the cases where y is a categorical variable
        if y is not int: y.to_numpy()
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])

        # end TODO
        return entropy
        
    def _create_split(self, X, thresh):
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    criterion_function = {'gini': _gini, 'entropy': _entropy}
    def _information_gain(self, X, y, thresh, criterion_function):
        # TODO: fix the code so it can switch between the two criterion: gini and entropy 
        if criterion_function == 'entropy':
            parent_loss = self._entropy(y)
            left_idx, right_idx = self._create_split(X, thresh)
            n, n_left, n_right = len(y), len(left_idx), len(right_idx)

            if n_left == 0 or n_right == 0: 
                return 0
        
            child_loss = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
        # end TODO
            return parent_loss - child_loss
        else:
            parent_loss = self._gini(y)
            left_idx, right_idx = self._create_split(X, thresh)
            n, n_left, n_right = len(y), len(left_idx), len(right_idx)

            if n_left == 0 or n_right == 0: 
                return 0
        
            child_loss = (n_left / n) * self._gini(y[left_idx]) + (n_right / n) * self._gini(y[right_idx])
        # end TODO
            return parent_loss - child_loss



    def _best_split(self, X, y, features):
        '''TODO: add comments here

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
        '''TODO: add some comments here
        '''
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestModel(object):

    def __init__(self, n_estimators):
        # TODO:
        pass
        self.n_estimators = n_estimators    # Number of trees
        # Initialize decision trees
        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(
                DecisionTreeModel())
        # end TODO

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # TODO:
        pass
        n_features = np.shape(X)[1]
        # If max_features have not been defined => select it as
        # sqrt(n_features)
        if not self.max_features:
            self.max_features = int(math.sqrt(n_features))

        # Choose one random subset of the data for each tree
            
        subsets = get_random_subsets(X, y, self.n_estimators)

        for i in self.progressbar(range(self.n_estimators)):
            X_subset, y_subset = subsets[i]
            # Feature bagging (select random subsets of the features)
            idx = np.random.choice(range(n_features), size=self.max_features, replace=True)
            # Save the indices of the features for prediction
            self.trees[i].feature_indices = idx
            # Choose the features corresponding to the indices
            X_subset = X_subset[:, idx]
            # Fit the tree to the data
            self.trees[i].fit(X_subset, y_subset)
        # end TODO
        pass


    def predict(self, X: pd.DataFrame):
        # TODO:
        pass
        y_preds = np.empty((X.shape[0], len(self.trees)))
        # Let each tree make a prediction on the data
        for i, tree in enumerate(self.trees):
            # Indices of the features that the tree has trained on
            idx = tree.feature_indices
            # Make a prediction based on those features
            prediction = tree.predict(X[:, idx])
            y_preds[:, i] = prediction
            
        y_pred = []
        # For each sample
        for sample_predictions in y_preds:
            # Select the most common class prediction
            y_pred.append(np.bincount(sample_predictions.astype('int')).argmax())
        return y_pred
        # end TODO

def get_random_subsets(X, y, n_subsets, replacements=True):
    """ Return random subsets (with replacements) of the data """
    n_samples = np.shape(X)[0]
    # Concatenate x and y and do a random shuffle
    X_y = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)
    np.random.shuffle(X_y)
    subsets = []

    # Uses 50% of training samples without replacements
    subsample_size = int(n_samples // 2)
    if replacements:
        subsample_size = n_samples      # 100% with replacements

    for _ in range(n_subsets):
        idx = np.random.choice(
            range(n_samples),
            size=np.shape(range(subsample_size)),
            replace=replacements)
        X = X_y[idx][:, :-1]
        y = X_y[idx][:, -1]
        subsets.append([X, y])
    return subsets
    

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
    
def classification_report(y_test, y_pred):
    # calculate precision, recall, f1-score

    TP = y_test == 1 & y_pred == 1
    FP = y_test == 0 & y_pred == 1
    FN = y_test == 1 & y_pred == 0

    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    f1_score = (2 * (precision * recall)) / (precision + recall)  #my code 
    # TODO:
    result = precision, recall, f1_score
    # end TODO
    return(result)

def confusion_matrix(y_test, y_pred):
    # return the 2x2 matrix
    # TODO:
    print(classification_report(y_test, y_pred))  #my code 
    result = np.array([[0, 0], [0, 0]])
    # end TODO
    return(result)

#def classification_report(y_test, y_pred):
    # calculate precision, recall, f1-score
    # TODO:
    #result = 'To be implemented'
    # end TODO
    #return(result)

#def confusion_matrix(y_test, y_pred):
    # return the 2x2 matrix
    # TODO:
    #result = np.array([[0, 0], [0, 0]])
    # end TODO
    #return(result)


def _test():
    
    df = pd.read_csv('breast_cancer.csv')
    
    #X = df.drop(['diagnosis'], axis=1).to_numpy()
    #y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1).to_numpy()

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
