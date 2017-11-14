import pandas as pd
import numpy as np

from sklearn import datasets, preprocessing, metrics
from ..models.mob import GLMTreeClassifier, GLMForestClassifier

data = datasets.load_breast_cancer()
X = pd.DataFrame(preprocessing.scale(data.data), columns=data.feature_names)
y = data.target

def test_glmtree():
	model = GLMTreeClassifier(reg_columns=['mean radius', 'mean perimeter'], factor_columns=['worst compactness']).fit(X, y)
	pred = model.predict(X)
	print(metrics.mean_squared_error(y, pred))

def test_glmforest():
	model = GLMForestClassifier(verbose=2, n_estimators=1, reg_columns=['mean radius', 'mean perimeter'], factor_columns=['worst compactness']).fit(X, y)
	pred = model.predict(X)
	print(metrics.mean_squared_error(y, pred))

if __name__ == '__main__':
	test_glmforest()
	test_glmtree()