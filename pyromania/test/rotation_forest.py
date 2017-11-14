#!/usr/bin/env python3

import pandas as pd
import numpy as np

from sklearn import datasets, preprocessing, metrics
from ..models.rotation_forest import RotationForestClassifier

def test():
	data = datasets.load_breast_cancer()
	X = pd.DataFrame(preprocessing.scale(data.data), columns=data.feature_names)
	y = data.target

	model = RotationForestClassifier().fit(X, y)
	pred = model.predict(X)
	print(metrics.mean_squared_error(y, pred))

if __name__ == '__main__':
	test()