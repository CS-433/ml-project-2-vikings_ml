import numpy as np

def error_rate(predictions, labels):
    """(ETH) Return the error rate based on dense predictions and 1-hot labels.
    
    Parameters
    ----------
    predictions: ndarray
        Numpy array with predictions
    labels: ndarray
        Numpy array with true labels
    
    returns
    The error rate
    """
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
        predictions.shape[0])