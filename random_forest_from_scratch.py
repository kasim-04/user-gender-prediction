import numpy as np
from sklearn.preprocessing import LabelEncoder


class Node:

    def __init__(self, prob, feature=None, threshold=None, left=None, right=None):
        self.prob = prob
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self._check_invariants()


    def is_leaf(self):
        return self.left is None and self.right is None


    def _check_invariants(self):
        if not isinstance(self.prob, (int, float)):
            raise TypeError(f'prob must be numeric, got {type(self.prob)}')

        if not (0.0 <= self.prob <= 1.0):
            raise ValueError(f'prob must be in [0, 1], got {self.prob}')

        if self.feature is not None and not isinstance(self.feature, int):
            raise TypeError(f'feature must be integer or None, got {type(self.feature)}')

        if self.threshold is not None and not isinstance(self.threshold, (int, float)):
            raise TypeError(f'threshold must be numeric or None, got {type(self.threshold)}')

        if self.left is not None and not isinstance(self.left, Node):
            raise TypeError(f'left must be Node or None, got {type(self.left)}')

        if self.right is not None and not isinstance(self.right, Node):
            raise TypeError(f'right must be Node or None, got {type(self.right)}')


class BinaryDecisionTreeClassifier:

    def __init__(self, max_depth, min_samples_split, min_samples_leaf, n_bins):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins

        self.thresholds_ = None
        self.n_features_ = None
        self.tree_ = None

        self._check_invariants()


    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.thresholds_ = self._get_thresholds(X)
        self.tree_ = self._grow_tree(X, y)

        return self


    def predict_proba(self, X):
        return np.array([self._traverse(x_i).prob for x_i in X])


    def predict(self, X, threshold=0.5):
        if not isinstance(threshold, (int, float)):
            raise TypeError(f'prob must be numeric, got {type(threshold)}')

        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f'threshold must be in [0, 1], got {threshold}')

        y_prob = self.predict_proba(X)
        y_pred = (y_prob >= threshold).astype(int)
        
        return y_pred


    def _check_invariants(self):
        if not isinstance(self.max_depth, int):
            raise TypeError(f'max_depth must be integer, got {type(self.max_depth)}')

        if self.max_depth <= 0:
            raise ValueError(f'max_depth must be positive, got {self.max_depth}')

        if not isinstance(self.min_samples_split, int):
            raise TypeError(f'min_samples_split must be integer, got {type(self.min_samples_split)}')

        if self.min_samples_split < 2:
            raise ValueError(f'min_samples_split must be >= 2, got {self.min_samples_split}')

        if not isinstance(self.min_samples_leaf, int):
            raise TypeError(f'min_samples_leaf must be integer, got {type(self.min_samples_leaf)}')

        if self.min_samples_leaf < 1:
            raise ValueError(f'min_samples_leaf must be >= 1, got {self.min_samples_leaf}')

        if not isinstance(self.n_bins, int):
            raise TypeError(f'n_bins must be integer, got {type(self.n_bins)}')

        if self.n_bins < 2:
            raise ValueError(f'n_bins must be >= 2, got {self.n_bins}')


    def _get_thresholds(self, X):
        thresholds = []

        for j in range(self.n_features_):
            x_j = X[:, j]
            min_val = x_j.min()
            max_val = x_j.max()

            if min_val == max_val:
                t = np.array([min_val])
            else:
                t = np.linspace(min_val, max_val, self.n_bins + 1)[1:-1]

            thresholds.append(t)

        return thresholds


    def _gini(self, y):
        probabilities = np.bincount(y, minlength=2) / len(y)
        return 1 - np.sum(probabilities ** 2)


    def _information_gain(self, y_parent, y_left, y_right, gini_parent):
        n_parent = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)

        gini_left = self._gini(y_left)
        gini_right = self._gini(y_right)

        return gini_parent - (n_left / n_parent * gini_left + n_right / n_parent * gini_right)


    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        gini = self._gini(y)
        features = np.random.choice(self.n_features_, size=int(np.sqrt(self.n_features_)) + 1, replace=False)

        for feature in features:
            values = X[:, feature]
    
            for threshold in self.thresholds_[feature]:
                mask = (values <= threshold)
                y_left = y[mask]
                y_right = y[~mask]

                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                gain = self._information_gain(y, y_left, y_right, gini)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold


    def _grow_tree(self, X, y, depth=0):
        prob = np.mean(y)

        if depth >= self.max_depth or len(y) < self.min_samples_split or np.all(y == y[0]):
            return Node(prob=prob)

        feature, threshold = self._best_split(X, y)

        if feature is None or threshold is None:
            return Node(prob=prob)

        mask = (X[:, feature] <= threshold)
        left = self._grow_tree(X[mask], y[mask], depth + 1)
        right = self._grow_tree(X[~mask], y[~mask], depth + 1)

        return Node(prob=prob, feature=feature, threshold=threshold, left=left, right=right)


    def _traverse(self, X_i):
        node = self.tree_

        while not node.is_leaf():
            if X_i[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node


class BinaryRandomForestClassifier:

    def __init__(self, n_estimators, max_depth, min_samples_split, min_samples_leaf, n_bins):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins

        self.label_encoder_ = None
        self.n_features_ = None
        self.forest_ = None

        self._check_invariants()


    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(y)
        y = np.asarray(y, dtype=int)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f'Number of samples in X ({X.shape[0]}) does not match number of samples in y ({y.shape[0]})')

        n_classes = len(np.unique(y))
        if n_classes > 2:
            raise ValueError(f'This BinaryDecisionTreeClassifier supports only 2 classes, got {n_classes}')

        n_samples, self.n_features_ = X.shape
        self.forest_ = []

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            tree = BinaryDecisionTreeClassifier(self.max_depth, self.min_samples_split, self.min_samples_leaf, self.n_bins)
            tree.fit(X_boot, y_boot)
            self.forest_.append(tree)

        return self


    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_:
            raise ValueError(f'Expected {self.n_features_} features, got {X.shape[1]}')

        probabilities = np.array([tree.predict_proba(X) for tree in self.forest_])
        return np.mean(probabilities, axis=0)


    def predict(self, X, threshold=0.5):
        if not isinstance(threshold, (int, float)):
            raise TypeError(f'threshold must be numeric, got {type(threshold)}')
        
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f'threshold must be in [0, 1], got {threshold}')

        y_prob = self.predict_proba(X)
        y_pred = (y_prob >= threshold).astype(int)
        
        return self.label_encoder_.inverse_transform(y_pred)


    def _check_invariants(self):
        if not isinstance(self.n_estimators, int):
            raise TypeError(f'n_estimators must be integer, got {type(self.n_estimators)}')

        if self.n_estimators <= 0:
            raise ValueError(f'n_estimators must be positive, got {self.n_estimators}')

        if not isinstance(self.max_depth, int):
            raise TypeError(f'max_depth must be integer, got {type(self.max_depth)}')

        if self.max_depth <= 0:
            raise ValueError(f'max_depth must be positive, got {self.max_depth}')

        if not isinstance(self.min_samples_split, int):
            raise TypeError(f'min_samples_split must be integer, got {type(self.min_samples_split)}')

        if self.min_samples_split < 2:
            raise ValueError(f'min_samples_split must be >= 2, got {self.min_samples_split}')

        if not isinstance(self.min_samples_leaf, int):
            raise TypeError(f'min_samples_leaf must be integer, got {type(self.min_samples_leaf)}')

        if self.min_samples_leaf < 1:
            raise ValueError(f'min_samples_leaf must be >= 1, got {self.min_samples_leaf}')

        if not isinstance(self.n_bins, int):
            raise TypeError(f'n_bins must be integer, got {type(self.n_bins)}')

        if self.n_bins < 2:
            raise ValueError(f'n_bins must be >= 2, got {self.n_bins}')