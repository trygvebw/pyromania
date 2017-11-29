import pandas as pd
import numpy as np

from sklearn import datasets, preprocessing, metrics, model_selection
from ..glm.glmnet import GLMNet

def test_gaussian_glmnet():
	data = datasets.load_diabetes()
	X_train, X_valid, y_train, y_valid = model_selection.train_test_split(preprocessing.scale(data.data), data.target)

	X_train = pd.DataFrame(X_train, columns=data.feature_names)
	X_valid = pd.DataFrame(X_valid, columns=data.feature_names)

	model = GLMNet().fit(X_train, y_train)
	pred = model.predict(X_valid)

	print(metrics.mean_squared_error(y_valid, pred))

if __name__ == '__main__':
	test_gaussian_glmnet()