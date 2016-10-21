import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

filename = "MarshOrchid.jpg"
image = mpimg.imread(filename)

x = tf.Variable(image, name="x")

model = tf.initialize_all_variables()

with tf.Session() as session:
    shape = tf.shape(x)
    session.run(model)
    shape = session.run(shape)
    print(shape)

    left_side = tf.slice(x, [0, 0, 0], [-1, shape[1]/2, -1])
    shape = tf.shape(left_side)
    session.run(model)
    shape = session.run(shape)
    print(shape)

    mirrored_left = tf.reverse_sequence(left_side, np.ones((shape[0],))*shape[1], 1, batch_dim=0)
    mirrored = tf.concat(1, [left_side, mirrored_left])
    session.run(model)
    result = session.run(mirrored)

print(result.shape)
plt.imshow(result)
plt.show()
