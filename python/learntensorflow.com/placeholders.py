import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


filename = "MarshOrchid.jpg"
raw_image_data = mpimg.imread(filename)
height, width, _ = raw_image_data.shape

image = tf.placeholder(tf.uint8, [None, None, 3])
topleft = tf.slice(image, [0, 0, 0], [height//2, width//2, -1])
bottomleft = tf.slice(image, [height//2, 0, 0], [-1, width//2, -1])
topright = tf.slice(image, [0, width//2, 0], [height//2, -1, -1])
bottomright = tf.slice(image, [height//2, width//2, 0], [-1, -1, -1])
left = tf.concat(0, [topleft, bottomleft])
right = tf.concat(0, [topright, bottomright])
merged = tf.concat(1, [left, right])
greyscale = tf.reduce_mean(image, 2)

with tf.Session() as session:
    result = session.run(topleft, feed_dict={image: raw_image_data})
    plt.imshow(result)
    plt.show()
    result = session.run(bottomleft, feed_dict={image: raw_image_data})
    plt.imshow(result)
    plt.show()
    result = session.run(topright, feed_dict={image: raw_image_data})
    plt.imshow(result)
    plt.show()
    result = session.run(bottomright, feed_dict={image: raw_image_data})
    plt.imshow(result)
    plt.show()
    result = session.run(merged, feed_dict={image: raw_image_data})
    plt.imshow(result)
    plt.show()
    result = session.run(greyscale, feed_dict={image: raw_image_data})
    plt.imshow(result, cmap="gray")
    plt.show()
