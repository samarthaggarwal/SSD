import tensorflow as tf
import ipdb

def cce_loss(true, pred):
    # cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
    # cce = tf.keras.losses.CategoricalCrossentropy()
    # return cce(true, pred)
    return tf.losses.softmax_cross_entropy(true, pred)

def combined_loss(true, pred):
    alpha = 1
    confidence_loss = tf.losses.softmax_cross_entropy(true[:,:,:11], pred[:,:,:11])
    # confidence_loss = cce_loss(true[:,:,:11], pred[:,:,:11])
    location_loss = tf.losses.huber_loss(true[:,:,11:15], pred[:,:,11:15])
    return confidence_loss + alpha * location_loss

def total_loss(true, pred):
    
    # mask = tf.equal(true[:,:,0:1], tf.constant(1.0))
    mask = tf.greater_equal(true[:,:,0:1], tf.constant(0.99))
    # neg_mask = tf.equal(true[:,:,0:1], tf.constant(0.0))
    neg_mask = tf.less_equal(true[:,:,0:1], tf.constant(0.01))
    mask = tf.cast(mask, tf.float32)
    neg_mask = tf.cast(neg_mask, tf.float32)

    loss_1 = combined_loss(true, pred)
    loss_2 = cce_loss(true[:,:,0:1], pred[:,:,0:1])
    loss = tf.add(tf.multiply(mask, loss_1) , tf.multiply(neg_mask, loss_2) )
    return tf.reduce_sum(loss)

"""
mask*loss_1 + (1-mask)*loss_2 = loss_2 + mask * (loss_1-loss_2)
"""
    

if __name__ == "__main__":
    pass

