import math
import pandas as pd
import numpy as np

import rpy2.robjects as robj
from rpy2.robjects import pandas2ri
import rpy2.robjects.packages as rpack
import rpy2.robjects.numpy2ri

from sklearn import datasets, preprocessing, metrics

rpy2.robjects.numpy2ri.activate()
r = robj.r

def install_package(package_name):
	utils = rpack.importr('utils')
	utils.chooseCRANmirror(ind=1)
	if not rpack.isinstalled(package_name):
		utils.install_packages(robj.vectors.StrVector([package_name]))

def fix_types(X, y=None, factor_columns=None, start_indexing_at=0):
	if type(X) is not pd.DataFrame:
		X = pd.DataFrame(X, columns=['X' + str(i) for i in range(start_indexing_at, X.shape[1] + start_indexing_at)])
	if factor_columns is not None:
		for col in factor_columns:
			X[col] = X[col].astype(str)
	if y is not None and type(y) is pd.DataFrame:
		y = y.values
	with robj.conversion.localconverter(robj.default_converter + pandas2ri.converter) as cv:
		X_r = pandas2ri.DataFrame(X)
	if y is None:
		return X_r, X
	else:
		return X_r, X, y

def convert_to_numpy_columns(X, columns):
	out = []
	X_cols = list(X)
	for col in columns:
		out.append('X' + str(X_cols.index(col) + 1))
	return out

def import_or_install(package_name):
	try:
		rpack.importr(package_name)
	except Exception as e:
		print('Installing package ' + package_name)
		install_package(package_name)
		rpack.importr(package_name)