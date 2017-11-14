import math
import pandas as pd
import numpy as np

import rpy2.robjects as robj
import rpy2.robjects.packages as rpack
r = robj.r

from sklearn import datasets, preprocessing, metrics
from ..util.data import install_package, fix_types

class RotationForestClassifier:
	def __init__(self, n_estimators=10, max_features=0.5, verbose=False):
		self.model = None
		self.ncol = None
		self.n_estimators = n_estimators
		self.max_features = max_features
		self.verbose = verbose

		install_package('rotationForest')
		rpack.importr('rotationForest')

	def fit(self, X, y):
		self.ncol = X.shape[1]
		X_r, X, y = fix_types(X, y)
		self.model = r.rotationForest(X_r, r.factor(y), K=int(math.floor(1/self.max_features)), L=self.n_estimators, verbose=self.verbose)
		return self

	def predict_proba(self, X):
		X_r, X = fix_types(X)
		return np.array(r.predict(self.model, X_r))

	def predict(self, X, threshold=0.5):
		proba = self.predict_proba(X)
		return (proba > threshold).astype(int)

	def news(self):
		print(r.rotationForestNews())