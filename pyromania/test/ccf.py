import pandas as pd
import numpy as np

from sklearn import datasets, preprocessing, metrics, model_selection
from ..models.ccf import CanonicalCorrelationForestClassifier

def test_ccf():
	data = datasets.load_breast_cancer()
	X_train, X_valid, y_train, y_valid = model_selection.train_test_split(preprocessing.scale(data.data), data.target)

	X_train = pd.DataFrame(X_train, columns=data.feature_names)
	X_valid = pd.DataFrame(X_valid, columns=data.feature_names)

	model = CanonicalCorrelationForestClassifier(n_estimators=50, projection_bootstrap=False).fit(X_train, y_train)
	pred = model.predict(X_valid)
	print(metrics.mean_squared_error(y_valid, pred))

if __name__ == '__main__':
	test_ccf()