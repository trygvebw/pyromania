import math
import pandas as pd
import numpy as np

import rpy2.robjects as robj
from rpy2.robjects import pandas2ri
import rpy2.robjects.packages as rpack
import rpy2.robjects.numpy2ri

from sklearn import datasets, preprocessing, metrics
from ..util.data import install_package, fix_types

rpy2.robjects.numpy2ri.activate()
r = robj.r

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
		X, y = fix_types(X, y)
		self.model = r.rotationForest(X, r.factor(y), K=int(math.floor(1/self.max_features)), L=self.n_estimators, verbose=self.verbose)
		return self

	def predict(self, X):
		X = fix_types(X)
		return np.array(r.predict(self.model, X))