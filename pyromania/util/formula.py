def additive(y_cols, X_reg_cols, X_partition_cols):
	if len(X_reg_cols) == 0:
		return '{} ~ {}'.format(' + '.join(y_cols), ' + '.join(X_partition_cols))
	else:
		return '{} ~ {} | {}'.format(' + '.join(y_cols), ' + '.join(X_reg_cols), ' + '.join(X_partition_cols))

def sanitize_column_names(names):
	return [s.replace(' ', '__') for s in names]