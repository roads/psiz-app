import tensorflow as tf
import numpy as np

def project_attention_weights(attention_weights_0):
    '''Projection of attention weights for projected gradient descent.
    '''
    n_dim = tf.shape(attention_weights_0, out_type=tf.float64)[1]
    attention_weights_1 = tf.divide(tf.reduce_sum(attention_weights_0, axis=1, keep_dims=True), n_dim)
    attention_weights_proj = tf.divide(attention_weights_0, attention_weights_1)
    
    return attention_weights_proj

def main():
    '''
    '''
    attention_weights = np.array([[1., 1., 1., 1.], [1., 2., 1.5, 1.]])
    

    attention_weights_0 = tf.constant(attention_weights)
    attention_weights_1 = project_attention_weights(attention_weights_0)

    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    sess = tf.InteractiveSession()
    print('attention_weights_0 = ', sess.run(attention_weights_0))
    print('attention_weights_1 = ', sess.run(attention_weights_1))
    sess.close()

    # # One group
    # n_reference = np.ones(5)*2
    # group_id = np.zeros(5)
    # attention_weights = np.ones((1, 3), dtype=np.float32)
    
    # # Two groups
    # # n_reference = np.ones(10)*2
    # # group_id = np.hstack((np.zeros(5), np.ones(5)))
    # # attention_weights = np.vstack((1.0*np.ones((1, 3)), 2.0*np.ones((1, 3))))
    
    # tf_attention_weights = tf.constant(attention_weights)

    # tf_n_reference = tf.constant(n_reference, dtype=tf.int32)
    # tf_group_id = tf.constant(group_id, dtype=tf.int32)

    # idx_2c1 = tf.squeeze(tf.where(tf.equal(tf_n_reference, tf.constant(2))))

    # group_idx = tf.gather(tf_group_id, idx_2c1)
    # # group_idx_t = tf.transpose([group_idx])
    # group_idx_t = tf.reshape(group_idx, [tf.shape(group_idx)[0],1])
    # weights_2c1 = tf.gather_nd(tf_attention_weights, group_idx_t)   

    # init = tf.global_variables_initializer()
    # sess = tf.Session()
    # sess.run(init)

    # sess = tf.InteractiveSession()
    # w = sess.run(weights_2c1)
    # print('weights= ', w)
    # sess.close()

if __name__ == "__main__":
    main()