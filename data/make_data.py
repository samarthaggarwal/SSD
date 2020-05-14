import pickle
# import numpy as np
# import tensorflow as tf


# input image shape : (734x820x3)
# format : (class_id, b_x, b_y, b_w, b_h)
labels = [
            [(1,500,600,40,50), (3, 200, 300, 50, 60)]
        ]
# labels_np = np.asarray(labels, np.float32)
# labels_tf = tf.convert_to_tensor(labels_np, np.float32)

label_path = 'labels.pkl'

with open(label_path, 'wb') as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('saved labels to label_path')