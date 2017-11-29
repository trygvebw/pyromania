import os
import math
import sys
import pandas as pd
import numpy as np

import rpy2.robjects as robj
import rpy2.robjects.packages as rpack
r = robj.r
STM = robj.functions.SignatureTranslatedFunction

from sklearn import datasets, preprocessing, metrics, ensemble
from ..util.data import install_package, fix_types, convert_to_numpy_columns, import_or_install
from ..util.formula import additive, sanitize_column_names

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

ALLOWED_FAMILIES = ('gaussian', 'binomial', 'poisson')

NAN_ACTIONS_TO_R = {
	'fail': 'na.fail',
	'omit': 'na.omit',
	'exclude': 'na.exclude',
	'pass': 'na.pass'
}

class GLMNet:
	def __init__(self, family='gaussian', alpha=1, fit_intercept=True, nan_action='fail',
			epsilon=1e-7, max_iter=100000, embed_data=False, verbose=False):
		# TODO: Handle multinomial and multigaussian families as automatic cases of binomial and gaussian
		self._verify_constructor_arguments(family, nan_action)

		self.family = family
		self.alpha = alpha
		self.fit_intercept = fit_intercept
		self.nan_action = nan_action
		self.epsilon = epsilon
		self.max_iter = max_iter
		self.embed_data = embed_data
		self.verbose = verbose

		self.model = None
		self._utils = import_or_install('glmnetUtils')

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

		self.model = STM(self._utils.cv_glmnet, init_prm_translate={'use_model_frame': 'use.model.frame', 'nan_action': 'na.action'})(
			formula, data=X_r, alpha=self.alpha, family=self.family, nan_action=NAN_ACTIONS_TO_R[self.nan_action],
			intercept=self.fit_intercept, thresh=self.epsilon, maxit=self.max_iter
		)
		self.model.rclass = robj.StrVector(('cv.glmnet.formula', 'cv.glmnet'))

		return self

	def predict(self, X, with_standard_errors=False, result_type='response'):
		if type(X) is pd.DataFrame:
			X.columns = sanitize_column_names(list(X))
		X_r, X = fix_types(X)
		if with_standard_errors:
			prediction_object = self._utils['predict.glmnet'](self.model, X_r, se_fit=True, type=result_type)
			return np.array(prediction_object['fit']), np.array(prediction_object['se.fit']), float(prediction_object['residual.scale'])
		else:
			return np.array(r['predict'](self.model, newdata=X_r, type=result_type, s=self.model.rx2('lambda.min')))

	def _verify_constructor_arguments(self, family, nan_action):
		assert family in ALLOWED_FAMILIES
		assert nan_action in NAN_ACTIONS_TO_R.keys()

	def _categorical_target(self):
		# TODO: Add quasibinomial
		return self.family == 'binomial'