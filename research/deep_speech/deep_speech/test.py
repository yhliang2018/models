import tensorflow as tf
import numpy as np
import math

# a = tf.constant([1.0, 2.0])
# b = tf.constant([3.0])

# c = [a, b]
# print(c)

# ten = tf.convert_to_tensor(c)

# print(ten)

data = [
  np.array([1, 2, 3, 4], dtype=np.int8),
  np.array([1, 2, 4], dtype=np.int8),
  np.array([1, 2, 3, 4], dtype=np.int8),
  np.array([3, 4], dtype=np.int8),
  np.array([1, 2, 3, 4], dtype=np.int8),
  np.array([1,], dtype=np.int8),
]

data0 = data[0]
data1 = np.expand_dims(data0, axis=1)
print(data1.shape)
print("here")

label = [
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


def _data_gen():
  for i in range(len(data)):
    yield (data[i], label[i])



g = tf.Graph()
with tf.Session(graph=g).as_default() as sess, g.as_default():
  batch_size = 4
  dataset = tf.data.Dataset.from_generator(generator=_data_gen, output_types=(tf.int8, tf.int8))
  dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None])))
     # tf.Dimension(None), tf.Dimension(None)))

  ds_it = dataset.make_one_shot_iterator()
  row = ds_it.get_next()
  for i in range(math.ceil(len(data)/4)):
    x = sess.run(row)
    # print(x)
    # if i % 10 == 0:
    print(i, x)
  print("done")

# print(len(str(g.as_graph_def()).encode("utf-8")))
# print(g.as_graph_def())
