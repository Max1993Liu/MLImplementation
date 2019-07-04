def mean_square_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def mean_absolute_error(y_true, y_pred):
	return (y_true, - y_pred).abs().mean()


