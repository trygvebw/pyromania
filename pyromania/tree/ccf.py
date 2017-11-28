import math
import pandas as pd
import numpy as np

import rpy2.robjects as robj
import rpy2.robjects.packages as rpack
r = robj.r

from sklearn import datasets, preprocessing, metrics
from ..util.data import install_package, fix_types, import_or_install_github

class CanonicalCorrelationForestClassifier:
	def __init__(self, n_estimators=10, verbose=False, projection_bootstrap=False):
		self.model = None
		self.ncol = None
		self.n_estimators = n_estimators
		self.verbose = verbose
		self.projection_bootstrap = projection_bootstrap

		import_or_install_github('ccf', 'jandob/ccf')

	def fit(self, X, y):
		self.ncol = X.shape[1]
		X_r, X, y = fix_types(X, y)
		self.model = r.canonical_correlation_forest(X_r, r.factor(y),
			ntree=self.n_estimators, verbose=self.verbose, projectionBootstrap=self.projection_bootstrap)
		return self

	def predict(self, X, threshold=0.5):
		X_r, X = fix_types(X)
		return np.array(r.predict(self.model, X_r)).astype(int)-1