import tensorflow as tf

def weighted_mean_squared_error(y_true,
                                y_pred,
                                w=0.1):
    y_pred_of_nonzeros = tf.where(tf.equal(y_true, 0),
                                  tf.subtract(y_pred, y_pred),
                                  y_pred)

    # if y_true is non-zero and y_pred is zero => this error will cost the model (w*100)% more than any other error
    return tf.reduce_mean(tf.square(y_true - y_pred_of_nonzeros)) + tf.reduce_mean(tf.square(y_true - y_pred)) * w