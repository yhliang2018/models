import tensorflow as tf
import numpy as np
import math

# a = tf.constant([1.0, 2.0])
# b = tf.constant([3.0])

# c = [a, b]
# print(c)

# ten = tf.convert_to_tensor(c)

# print(ten)
a = [np.array([[1,2,3], [4,5,6]])]

data = [
  np.array([[1, 2, 3, 4], [1,2,3,4]], dtype=np.int8),
  np.array([[1, 2, 3, 4], [1,2,3,4], [1,2,3,4]], dtype=np.int8),
  np.array([[1, 2, 3, 4]], dtype=np.int8),
  np.array([[1, 2, 3, 4], [1,2,3,4]], dtype=np.int8),
  np.array([[1, 2, 3, 4], [1,2,3,4], [1,2,3,4], [1,2,3,4]],dtype=np.int8),
  np.array([[1, 2, 3, 4], [1,2,3,4]], dtype=np.int8),
]

data0 = data[0]
data1 = np.expand_dims(data0, axis=2)
print(data1.shape)
print("here")

labels = [
  np.array([1, 2], dtype=np.int8),
  np.array([1, 2, 3, 4], dtype=np.int8),
  np.array([1,], dtype=np.int8),
  np.array([3], dtype=np.int8),
  np.array([1, 2], dtype=np.int8),
  np.array([1,], dtype=np.int8),
]

# k = 1000
# for _ in range(k):
#   data.append(np.array([1,2,3], dtype=np.int8))


# def _data_gen():
#   for i in range(len(data)):
#     yield (data[i], label[i])

def _data_gen():
  for i in range(len(data)):
    feature = np.expand_dims(data[i], axis=2)
    # feature = data[i]
    # input_length = np.expand_dims(feature.shape[0], axis=1)
    # label_length = np.expand_dims(len(labels[i]), axis=1)
    input_length = feature.shape[0]
    label_length = len(labels[i])
    yield ({
        "features": feature,
        "labels": labels[i],
        "input_length": input_length,
        "label_length": label_length
    })


g = tf.Graph()
with tf.Session(graph=g).as_default() as sess, g.as_default():
  batch_size = 2
  dataset = tf.data.Dataset.from_generator(generator=_data_gen, output_types = {
          "features": tf.float32,
          "labels": tf.int8,
          "input_length": tf.int8,
          "label_length": tf.int8
      })
  dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes={
          "features": (tf.Dimension(None), tf.Dimension(None), 1),
          "labels": tf.Dimension(None),
          "input_length": [],
          "label_length": []
      })
     # tf.Dimension(None), tf.Dimension(None)))

  ds_it = dataset.make_one_shot_iterator()
  row = ds_it.get_next()
  for i in range(math.ceil(len(data)/batch_size)):
    x = sess.run(row)
    # print(x)
    # if i % 10 == 0:
    print(i, x)
  print("done")

# print(len(str(g.as_graph_def()).encode("utf-8")))
# print(g.as_graph_def())
