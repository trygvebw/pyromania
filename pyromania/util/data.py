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

def fix_types(X, y=None):
	if type(X) is not pd.DataFrame:
			X = pd.DataFrame(X, columns=['X' + str(i) for i in range(X.shape[1])])
	if y is not None and type(y) is pd.DataFrame:
		y = y.values
	with robj.conversion.localconverter(robj.default_converter + pandas2ri.converter) as cv:
		X_r = pandas2ri.DataFrame(X)
	if y is None:
		return X_r
	else:
		return X_r, y