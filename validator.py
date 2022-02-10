import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf

def iou_numpy(y_true, y_pred):
    intersection = np.sum(np.multiply(y_true.astype('bool'), y_pred >= 0.5))
    union = np.sum((y_true.astype('bool') + y_pred.astype('bool')) > 0)
    return intersection / union

def iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU).
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the IoU for the given label
    """
    intersection = K.sum(y_true * K.round(y_pred))
    union = K.sum(y_true) + K.sum(K.round(y_pred)) - intersection
    return K.switch(K.equal(union, 0), 1.0, K.cast(intersection / union,tf.float32))
