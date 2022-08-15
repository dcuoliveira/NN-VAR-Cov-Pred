import tensorflow as tf

def weighted_mean_squared_error(y_true,
                                y_pred,
                                w=0.1):
    y_pred_of_nonzeros = tf.where(tf.equal(y_true, 0), y_pred - y_pred, y_pred)

    # non-zero target predictions are penalized less than the average prediction
    return tf.reduce_mean(tf.square(y_true - y_pred_of_nonzeros)) + tf.reduce_mean(tf.square(y_true - y_pred)) * w