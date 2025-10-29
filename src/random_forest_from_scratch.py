import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional, Tuple, List


class Node:
    """
    A node of a binary decision tree.

    Parameters
    ----------
    prob : float
        Probability of the positive class at this node.
    feature : int or None, default=None
        Feature index used for splitting. None if the node is a leaf.
    threshold : float or None, default=None
        Threshold value for splitting the feature. None if the node is a leaf.
    gain : float or None, default=None
        Information gain of the split. None for leaf nodes.
    left : Node or None, default=None
        Left child node (samples with feature values <= threshold).
    right : Node or None, default=None
        Right child node (samples with feature values > threshold).
    """
    def __init__(self,
                 prob: float,
                 feature: Optional[int] = None,
                 threshold: Optional[float] = None,
                 gain: Optional[float] = None,
                 left: Optional["Node"] = None,
                 right: Optional["Node"] = None) -> None:
        self.prob = prob
        self.feature = feature
        self.threshold = threshold
        self.gain = gain
        self.left = left
        self.right = right
        self._check_invariants()


    def is_leaf(self) -> bool:
        """
        Check if the node is a leaf.

        Returns
        -------
        is_leaf : bool
            True if the node has no children, False otherwise.
        """
        return self.left is None and self.right is None


    def _check_invariants(self) -> None:
        """
        Validate the node attributes for correct types and values.

        Raises
        ------
        ValueError
            If `prob` is not in [0, 1].
        """
        if not (0.0 <= self.prob <= 1.0):
            raise ValueError(f'prob must be in [0, 1], got {self.prob}')


class BinaryDecisionTreeClassifier:
    """
    Binary decision tree classifier.

    Parameters
    ----------
    max_depth : int, default=5
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    n_bins : int, default=16
        Number of bins to consider for thresholding features.
    """
    def __init__(self,
                 max_depth: int = 5,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 n_bins: int = 16) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins

        self.thresholds_ = None
        self.n_features_ = None
        self.tree_ = None

        self._check_invariants()


    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinaryDecisionTreeClassifier":
        """
        Fit the decision tree classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target vector.

        Returns
        -------
        self : BinaryDecisionTreeClassifier
            The fitted classifier.

        Raises
        ------
        ValueError
            If number of samples in X and y do not match or if there are more than 2 classes.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(f'Number of samples in X ({X.shape[0]}) does not match number of samples in y ({y.shape[0]})')

        n_classes = len(np.unique(y))
        if n_classes > 2:
            raise ValueError(f'This BinaryDecisionTreeClassifier supports only 2 classes, got {n_classes}')

        self.n_features_ = X.shape[1]
        self.thresholds_ = self._get_thresholds(X)
        self.tree_ = self._grow_tree(X, y)

        return self


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_prob : np.ndarray of shape (n_samples,)
            Probability for class 1.

        Raises
        ------
        ValueError
            If the number of features in X does not match the fitted model.
        """
        if X.shape[1] != self.n_features_:
            raise ValueError(f'Expected {self.n_features_} features, got {X.shape[1]}')

        return np.array([self._traverse(x_i).prob for x_i in X])


    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.
        threshold : float, default=0.5
            Probability threshold for predicting class 1.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        
        Raises
        ------
        ValueError
            If threshold is not in [0, 1].
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f'threshold must be in [0, 1], got {threshold}')

        y_prob = self.predict_proba(X)
        y_pred = (y_prob >= threshold).astype(int)

        return y_pred


    def compute_feature_importances(self) -> np.ndarray:
        """
        Compute feature importances based on information gain.

        Returns
        -------
        importances : np.ndarray of shape (n_features,)
            Normalized feature importances (sum to 1).
        """
        importances = np.zeros(self.n_features_)

        def collect(node):
            if node.is_leaf():
                return

            importances[node.feature] += node.gain
            collect(node.left)
            collect(node.right)

        collect(self.tree_)

        total = importances.sum()
        if total > 0:
            return importances / total

        return importances


    def _check_invariants(self):
        """
        Check validity of classifier parameters.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
        if self.max_depth <= 0:
            raise ValueError(f'max_depth must be positive, got {self.max_depth}')

        if self.min_samples_split < 2:
            raise ValueError(f'min_samples_split must be >= 2, got {self.min_samples_split}')

        if self.min_samples_leaf < 1:
            raise ValueError(f'min_samples_leaf must be >= 1, got {self.min_samples_leaf}')

        if self.n_bins < 2:
            raise ValueError(f'n_bins must be >= 2, got {self.n_bins}')


    def _get_thresholds(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Compute candidate thresholds for each feature.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        thresholds : List[np.ndarray]
            List of thresholds for each feature.
        """
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


    def _gini(self, y: np.ndarray) -> float:
        """
        Compute Gini impurity for a set of labels.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        gini : float
            Gini impurity.
        """
        probabilities = np.bincount(y, minlength=2) / len(y)
        return 1 - np.sum(probabilities ** 2)


    def _information_gain(self,
                          y_parent: np.ndarray,
                          y_left: np.ndarray,
                          y_right: np.ndarray,
                          gini_parent: float) -> float:
        """
        Compute information gain of a split.

        Parameters
        ----------
        y_parent : np.ndarray
            Labels of the parent node.
        y_left : np.ndarray
            Labels of the left child node.
        y_right : np.ndarray
            Labels of the right child node.
        gini_parent : float
            Gini impurity of the parent node.

        Returns
        -------
        gain : float
            Information gain of the split.
        """
        n_parent = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)

        gini_left = self._gini(y_left)
        gini_right = self._gini(y_right)

        return gini_parent - (n_left / n_parent * gini_left + n_right / n_parent * gini_right)


    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find the best feature and threshold to split the data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        best_feature : int or None
            Index of the best feature to split, or None if no split found.
        best_threshold : float or None
            Threshold value for the best split.
        best_gain : float
            Information gain for the best split.
        """
        best_gain = 0.0
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

        return best_feature, best_threshold, best_gain


    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> "Node":
        """
        Recursively grow a decision tree from data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target labels.
        depth : int, default=0
            Current depth of the tree.

        Returns
        -------
        Node
            Root node of the subtree.
        """
        prob = np.mean(y)

        if depth >= self.max_depth or len(y) < self.min_samples_split or np.all(y == y[0]):
            return Node(prob=prob)

        feature, threshold, gain = self._best_split(X, y)

        if feature is None or threshold is None or gain == 0.0:
            return Node(prob=prob)

        mask = (X[:, feature] <= threshold)
        left = self._grow_tree(X[mask], y[mask], depth + 1)
        right = self._grow_tree(X[~mask], y[~mask], depth + 1)

        return Node(prob=prob, feature=feature, threshold=threshold, gain=gain, left=left, right=right)


    def _traverse(self, X_i: np.ndarray) -> "Node":
        """
        Traverse the decision tree for a single sample to find the leaf node.

        Parameters
        ----------
        X_i : np.ndarray of shape (n_features,)
            Feature vector of a single sample.

        Returns
        -------
        Node
            Leaf node reached by traversing the tree according to the sample's feature values.
        """
        node = self.tree_

        while not node.is_leaf():
            if X_i[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node


class BinaryRandomForestClassifier(BaseEstimator, ClassifierMixin):
    """
    A simple binary random forest classifier using bootstrap aggregation of decision trees.

    Parameters
    ----------
    n_estimators : int, default=3
        Number of trees in the forest.
    max_depth : int, default=5
        Maximum depth of each decision tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    n_bins : int, default=16
        Number of bins used for threshold discretization in trees.
    """
    def __init__(self,
                 n_estimators: int = 3,
                 max_depth: int = 5,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 n_bins: int = 16) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins

        self.label_encoder_ = None
        self.n_features_ = None
        self.forest_ = None

        self._check_invariants()


    @property
    def feature_importances_(self):
        """
        Compute normalized feature importances across all trees in the forest.

        Returns
        -------
        importances : np.ndarray of shape (n_features,)
            Feature importances normalized to sum to 1.
        """
        importances = np.zeros(self.n_features_)
        for tree in self.forest_:
            importances += tree.compute_feature_importances()

        total = importances.sum()
        if total > 0:
            return importances / total

        return importances


    def fit(self, X: np.ndarray, y: np.ndarray) -> "BinaryRandomForestClassifier":
        """
        Fit the random forest classifier on the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Binary target values.

        Returns
        -------
        self : BinaryRandomForestClassifier
            The fitted model.

        Raises
        ------
        ValueError
            If number of samples in X and y do not match or if there are more than 2 classes.
        """
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


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X by averaging probabilities over all trees.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_prob : np.ndarray of shape (n_samples,)
            Probability for class 1.

        Raises
        ------
        ValueError
            If the number of features in X does not match the fitted model.
        """
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_:
            raise ValueError(f'Expected {self.n_features_} features, got {X.shape[1]}')

        probabilities = np.array([tree.predict_proba(X) for tree in self.forest_])
        y_prob = np.mean(probabilities, axis=0)

        return y_prob


    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.
        threshold : float, default=0.5
            Probability threshold for classifying as class 1.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.

        Raises
        ------
        ValueError
            If threshold is not in [0, 1].
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f'threshold must be in [0, 1], got {threshold}')

        y_prob = self.predict_proba(X)
        y_pred = (y_prob >= threshold).astype(int)

        return self.label_encoder_.inverse_transform(y_pred)


    def _check_invariants(self) -> None:
        """
        Validate hyperparameters for correct values.

        Raises
        ------
        ValueError
            If any hyperparameter violates constraints.
        """
        if self.n_estimators <= 0:
            raise ValueError(f'n_estimators must be positive, got {self.n_estimators}')

        if self.max_depth <= 0:
            raise ValueError(f'max_depth must be positive, got {self.max_depth}')

        if self.min_samples_split < 2:
            raise ValueError(f'min_samples_split must be >= 2, got {self.min_samples_split}')

        if self.min_samples_leaf < 1:
            raise ValueError(f'min_samples_leaf must be >= 1, got {self.min_samples_leaf}')

        if self.n_bins < 2:
            raise ValueError(f'n_bins must be >= 2, got {self.n_bins}')