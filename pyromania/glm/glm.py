import os
import math
import pandas as pd
import numpy as np

import rpy2.robjects as robj
import rpy2.robjects.packages as rpack
r = robj.r

from sklearn import datasets, preprocessing, metrics, ensemble
from ..util.data import install_package, fix_types, convert_to_numpy_columns, import_or_install
from ..util.formula import additive, sanitize_column_names

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

ALLOWED_LINK_FUNCTIONS = {
	'gaussian': ('identity', 'log', 'inverse'),
	'binomial': ('logit', 'probit', 'cauchit', 'log', 'cloglog'),
	'gamma': ('inverse', 'identity', 'log'),
	'poisson': ('log', 'identity', 'sqrt'),
	'inverse-gaussian': ('inverse-squared-mu', 'inverse', 'identity', 'log'),
	'quasi': ('identity', 'logit', 'probit', 'cloglog', 'inverse', 'log', 'inverse-squared-mu', 'sqrt'),
}

VARIANCE_OPTIONS = ('constant', 'mu(1-mu)', 'mu', 'mu^2', 'mu^3')

FAMILY_TO_R_FUNC = {**{a: a for a in ALLOWED_LINK_FUNCTIONS.keys()}, **{
	'gamma': 'Gamma',
	'inverse-gaussian': 'inverse_gaussian',
}}

class GLM:
	def __init__(self, family='gaussian', link=None, variance=None, fit_intercept=True, epsilon=1e-8, max_iter=25, verbose=False):
		self._verify_constructor_arguments(family, link, variance)

		self.family = family
		self.link = link if link is not None else ALLOWED_LINK_FUNCTIONS[family][0]
		self.variance = variance
		self.fit_intercept = fit_intercept
		self.epsilon = epsilon
		self.max_iter = max_iter
		self.verbose = verbose

		self.ncol = None
		self.model = None
		self.family_object = None

		#import_or_install('glm')

	def fit(self, X, y):
		self.ncol = X.shape[1]
		#X_r, X, y = fix_types(X, y)
		#self.model = r.rotationForest(X_r, r.factor(y), K=int(math.floor(1/self.max_features)), L=self.n_estimators, verbose=self.verbose)
		family_func = r[FAMILY_TO_R_FUNC[self.family]]
		if self.variance:
			self.family_object = family_func(link=self.link, variance=self.variance)
		else:
			self.family_object = family_func(link=self.link)

		target_column_name = 'target__'
		if type(X) is pd.DataFrame:
			X.columns = sanitize_column_names(list(X))
			X[target_column_name] = y
		else:
			y = y.reshape((-1, 1))
			X = np.concatenate((y, X), axis=1)
			target_column_name = 'X0'
		X_r, X = fix_types(X)
		formula = robj.Formula(additive([target_column_name], None, list(set(list(X)) - set([target_column_name]))))

		self.model = r.glm(formula, data=X_r, family=self.family_object,
			intercept=self.fit_intercept, control=r['glm.control'](maxit=self.max_iter, trace=self.verbose, epsilon=self.epsilon))
		self.model.rclass = robj.StrVector(('glm', 'lm'))
		return self

	def predict(self, X, with_standard_errors=False):
		if type(X) is pd.DataFrame:
			X.columns = sanitize_column_names(list(X))
		X_r, X = fix_types(X)
		if with_standard_errors:
			prediction_object = r.predict(self.model, X_r, se_fit=True)
			return np.array(prediction_object['fit']), np.array(prediction_object['se.fit']), float(prediction_object['residual.scale'])
		else:
			return np.array(r.predict(self.model, X_r))

	def _verify_constructor_arguments(self, family, link, variance):
		assert not (variance is not None and family != 'quasi')
		assert not (variance is not None and variance not in VARIANCE_OPTIONS)
		assert family in ALLOWED_LINK_FUNCTIONS.keys()
		assert link is None or link in ALLOWED_LINK_FUNCTIONS[family]