import math
import pandas as pd
import numpy as np

import rpy2.robjects as robj
import rpy2.robjects.packages as rpack
r = robj.r

from sklearn import datasets, preprocessing, metrics, ensemble
from ..util.data import install_package, fix_types, convert_to_numpy_columns, import_or_install
from ..util.formula import additive, sanitize_column_names

class GLMTree:
	def __init__(self, family='gaussian', reg_columns=None, factor_columns=None):
		self.model = None
		self.family = family
		self.reg_columns = reg_columns
		self.factor_columns = factor_columns
		self.random_state = 1 # FIXME: Make this do something
		self.x_is_df = False

		import_or_install('partykit')

	def fit(self, X, y, sample_weight=None, **kwargs):
		if self.random_state is not None:
			r('set.seed(' + str(self.random_state) + ')')
		reg_columns = self.reg_columns
		factor_columns = self.factor_columns
		if reg_columns is None:
			reg_columns = []
		else:
			reg_columns = sanitize_column_names(reg_columns)
		if factor_columns is None:
			factor_columns = []
		if sample_weight is None:
			sample_weight = [1]*X.shape[0]
		target_column_name = 'target__'
		if type(X) is pd.DataFrame:
			X.columns = sanitize_column_names(list(X))
			X[target_column_name] = y
		else:
			y = y.reshape((-1, 1))
			X = np.concatenate((y, X), axis=1)
			target_column_name = 'X0'
		self.factor_columns = sanitize_column_names(factor_columns)
		X_r, X = fix_types(X, factor_columns=self.factor_columns)
		formula = robj.Formula(additive([target_column_name], reg_columns, list(set(list(X)) - set([target_column_name]) - set(reg_columns))))

		self.model = r.glmtree(formula, data=X_r, family=self.family, weights=np.array(sample_weight))
		return self

	def predict_proba(self, X):
		X_r, X = fix_types(X, factor_columns=self.factor_columns, start_indexing_at=1)
		return np.array(r.predict(self.model, X_r, type='response'))

	def predict(self, X, threshold=0.5, **kwargs):
		proba = self.predict_proba(X)
		return (proba > threshold).astype(int)

	def get_params(self, deep=False):
		return {
			'reg_columns': self.reg_columns,
			'factor_columns': self.factor_columns,
		}

	def set_params(self, **kwargs):
		for arg in kwargs.keys():
			setattr(self, arg, kwargs[arg])

	def _validate_X_predict(self, X, **kwargs):
		return X

class GLMTreeClassifier(GLMTree):
	def __init__(self, **kwargs):
		super().__init__('binomial', **kwargs)

class GLMTreeRegressor(GLMTree):
	def __init__(self, **kwargs):
		super().__init__('gaussian', **kwargs)

class GLMForest(ensemble.forest.ForestRegressor):
	def __init__(self, family='gaussian', n_estimators=10, bootstrap=True, random_state=None, verbose=False, reg_columns=None, factor_columns=None):
		super().__init__(
			base_estimator=GLMTree(family),
			n_estimators=n_estimators,
			estimator_params=('family', 'reg_columns', 'factor_columns'),
			bootstrap=bootstrap,
			oob_score=False,
			n_jobs=1,
			random_state=random_state,
			verbose=verbose,
			warm_start=False)
		self.family = family
		self.reg_columns = reg_columns
		self.factor_columns = factor_columns

	def fit(self, X, y, *args, **kwargs):
		if type(X) is pd.DataFrame:
			self.reg_columns = convert_to_numpy_columns(X, self.reg_columns)
			self.factor_columns = convert_to_numpy_columns(X, self.factor_columns)
		if type(X) is pd.DataFrame:
			X = X.values
		if type(y) is pd.DataFrame:
			y = y.values
		return super().fit(X, y, *args, **kwargs)

	def predict(self, X, **kwargs):
		if type(X) is pd.DataFrame:
			X = X.values
		return super().predict(X, **kwargs)

	def predict_proba(self, X, **kwargs):
		if type(X) is pd.DataFrame:
			X = X.values
		return super().predict_proba(X, **kwargs)

class GLMForestRegressor(GLMForest):
	def __init__(self, **kwargs):
		super().__init__(family='gaussian', **kwargs)

class GLMForestClassifier(GLMForest):
	def __init__(self, **kwargs):
		super().__init__(family='binomial', **kwargs)

	def predict_proba(self, X):
		if type(X) is pd.DataFrame:
			X = X.values
		preds = []
		for est in self.estimators_:
			proba = est.predict_proba(X)
			preds.append(proba)
		return np.sum(preds, axis=0)/len(preds)

	def predict(self, X, threshold=0.5, **kwargs):
		proba = self.predict_proba(X)
		return (proba > threshold).astype(int)