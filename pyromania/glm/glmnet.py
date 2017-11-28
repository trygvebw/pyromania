import os
import math
import sys
import pandas as pd
import numpy as np

import rpy2.robjects as robj
import rpy2.robjects.packages as rpack
r = robj.r

from sklearn import datasets, preprocessing, metrics, ensemble
from ..util.data import install_package, fix_types, convert_to_numpy_columns, import_or_install
from ..util.formula import additive, sanitize_column_names

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

ALLOWED_FAMILIES = ('gaussian', 'binomial', 'poisson')

class GLMNet:
	def __init__(self, family='gaussian', alpha=1, fit_intercept=True,
			epsilon=1e-7, max_iter=100000, embed_data=False, verbose=False):
		self._verify_constructor_arguments(family, link, variance, nan_action)

		self.family = family
		self.alpha = alpha
		self.fit_intercept = fit_intercept
		self.epsilon = epsilon
		self.max_iter = max_iter
		self.embed_data = embed_data
		self.verbose = verbose

		self.model = None

		import_or_install_github('glmnetUtils')

	def fit(self, X, y):
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

		if self._categorical_target():
			X_r[X_r.colnames.index(target_column_name)] = robj.FactorVector(X_r.rx2(target_column_name))

		self.model = r.glmnet(formula, data=X_r, alpha=self.alpha, family=self.family_object, model=self.embed_data,
			y=self.embed_data, na_action=NAN_ACTIONS_TO_R[self.nan_action], intercept=self.fit_intercept,
			control=r['glm.control'](maxit=self.max_iter, trace=self.verbose, epsilon=self.epsilon))
		self.model.rclass = robj.StrVector(('glm', 'lm'))

		return self

	def predict(self, X, with_standard_errors=False, result_type='response'):
		if type(X) is pd.DataFrame:
			X.columns = sanitize_column_names(list(X))
		X_r, X = fix_types(X)
		if with_standard_errors:
			prediction_object = r.predict(self.model, X_r, se_fit=True, type=result_type)
			return np.array(prediction_object['fit']), np.array(prediction_object['se.fit']), float(prediction_object['residual.scale'])
		else:
			return np.array(r.predict(self.model, X_r, type=result_type))

	@property
	def info_(self):
		self._assert_model_is_fitted()
		model_value_dict = dict(self.model.items())
		scalar_fields = ('deviance', 'rank', 'aic', 'null.deviance',
			'iter', 'df.residual', 'df.null', 'converged', 'boundary')
		return {
			'coefficients': dict(self.model.rx2('coefficients').items()),
			**{s.replace('.', '_'): model_value_dict[s][0] for s in scalar_fields}
		}

	def _verify_constructor_arguments(self, family, link, variance, nan_action):
		assert not (variance is not None and family != 'quasi')
		assert not (variance is not None and variance not in VARIANCE_OPTIONS)
		assert family in ALLOWED_LINK_FUNCTIONS.keys()
		assert link is None or link in ALLOWED_LINK_FUNCTIONS[family]
		assert nan_action in NAN_ACTIONS_TO_R.keys()

	def _categorical_target(self):
		# TODO: Add quasibinomial
		return self.family == 'binomial'

	def _assert_model_is_fitted(self):
		if self.model is None:
			raise Exception('Fit the model before running {}'.format(sys._getframe(1).f_code.co_name))