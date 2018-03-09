# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools as it
import sys

import numpy as np
import resnet
import tensorflow as tf


class BlockTest(tf.test.TestCase):

  def generate_tests(self):
    """Automatically generate explicit tests.

      This function handles the tedious task of running various combinations
    of inputs and placing the results in a unit test.
    """

    batch_size = 32

    np.random.seed(68731974)

    for i in range(5):
      tf_seed = np.random.randint(0, 2**32-1)
      expected = self.dense_run(tf_seed)
      print("  def test_dense_{}(self):".format(i))
      print("    \"\"\"Sanity check {} on dense layer.\"\"\"".format(i))
      print("    computed = self.dense_run({})".format(tf_seed))
      print("    tf.assert_equal(computed, {})\n".format(expected))

    use_bottleneck = [True, False]
    use_projection = [True, False]
    input_widths = [4, 32, 128]
    input_channels = [4, 8, 32]

    block_permutations = it.product(*[use_bottleneck, use_projection,
                                      input_widths, input_channels])

    for bottleneck, projection, width, channels in block_permutations:
      tf_seed = np.random.randint(0, 2**32-1)
      expected_size, expected_values = self.resnet_block_run(
          tf_seed, batch_size=batch_size, bottleneck=bottleneck,
          projection=projection, width=width, channels=channels)
      block_name = "bottleneck" if bottleneck else "building"
      fn_name = "test_{}_block_width_{}_channels_{}_batch_size_{}{}".format(
          block_name, width, channels, batch_size,
          "_with_proj" if projection else "")

      print("  def {}(self):".format(fn_name))
      print("    \"\"\"Test of a single ResNet block.\"\"\"")
      tab_over = " "  * 8
      print("    computed_size, computed_values = self.resnet_block_run(\n"
            "{tab_over}{tf_seed}, batch_size={batch_size}, "
            "bottleneck={bottleneck}, projection={projection},\n{tab_over}"
            "width={width}, channels={channels})".format(
                tab_over=tab_over, tf_seed=tf_seed, batch_size=batch_size,
                bottleneck=bottleneck, projection=projection, width=width,
                channels=channels))
      print("    tf.assert_equal(computed_size, {})".format(expected_size))
      print("    tf.assert_equal(computed_values, {})".format(expected_values))
      print()

  def dense_run(self, tf_seed):
    """Simple generation of one random float and a single node dense network.

      The subsequent more involved tests depend on the ability to correctly seed
    TensorFlow. In the event that that process does not function as expected,
    the simple dense tests will fail indicating that the issue is with the
    tests rather than the ResNet functions.

    Args:
      tf_seed: Random seed for TensorFlow

    Returns:
      The generated random number and result of the dense network.
    """
    with self.test_session(graph=tf.Graph()) as sess:
      tf.set_random_seed(tf_seed)

      x = tf.random_uniform((1, 1))
      y = tf.layers.dense(inputs=x, units=1)

      init = tf.global_variables_initializer()
      sess.run(init)
      return x.eval()[0, 0], y.eval()[0, 0]

  def make_projection(self, filters_out, strides, data_format):
    """1D convolution with stride projector.

    Args:
      filters_out: Number of filters in the projection.
      strides: Stride length for convolution.
      data_format: channels_first or channels_last

    Returns:
      A 1 wide CNN projector function.
    """
    def projection_shortcut(inputs):
      return resnet.conv2d_fixed_padding(
          inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
          data_format=data_format)
    return projection_shortcut

  def resnet_block_run(self, tf_seed, batch_size, bottleneck, projection, width,
                       channels):
    """Test whether resnet block construction has changed.

      This function runs ResNet block construction under a variety of different
    conditions.

    Args:
      tf_seed: Random seed for TensorFlow
      batch_size: Number of points in the fake image. This is needed due to
        batch normalization.
      bottleneck: Whether or not to use bottleneck layers.
      projection: Whether or not to project the input.
      width: The width of the fake image.
      channels: The number of channels in the fake image.

    Returns:
      The size of the block output, as well as several check values.
    """
    data_format = "channels_last"

    try:
      block_fn = resnet._building_block_v2
      if bottleneck:
        block_fn = resnet._bottleneck_block_v2
    except:
      # TODO(taylorrobie): remove before the merge into master
      block_fn = resnet.building_block
      if bottleneck:
        block_fn = resnet.bottleneck_block

    with self.test_session(graph=tf.Graph()) as sess:
      tf.set_random_seed(tf_seed)

      strides = 1
      channels_out = channels
      projection_shortcut = None
      if projection:
        strides = 2
        channels_out *= strides
        projection_shortcut = self.make_projection(
            filters_out=channels_out, strides=strides, data_format=data_format)

      filters = channels_out
      if bottleneck:
        filters = channels_out // 4

      x = tf.random_uniform((batch_size, width, width, channels))

      y = block_fn(inputs=x, filters=filters, training=True,
                   projection_shortcut=projection_shortcut, strides=strides,
                   data_format=data_format)

      init = tf.global_variables_initializer()
      sess.run(init)

      y_array = y.eval()
      y_flat = y_array.flatten()
      return y_array.shape, (y_flat[0], y_flat[-1], np.sum(y_flat))

  #=============================================================================
  # Procedurally generated tests
  #=============================================================================
  def test_dense_0(self):
    """Sanity check 0 on dense layer."""
    computed = self.dense_run(1813835975)
    tf.assert_equal(computed, (0.8760674, 0.2547844))

  def test_dense_1(self):
    """Sanity check 1 on dense layer."""
    computed = self.dense_run(3574260356)
    tf.assert_equal(computed, (0.75590825, 0.5339718))

  def test_dense_2(self):
    """Sanity check 2 on dense layer."""
    computed = self.dense_run(599400476)
    tf.assert_equal(computed, (0.22491038, -0.02492056))

  def test_dense_3(self):
    """Sanity check 3 on dense layer."""
    computed = self.dense_run(309580726)
    tf.assert_equal(computed, (0.39424884, 0.17353162))

  def test_dense_4(self):
    """Sanity check 4 on dense layer."""
    computed = self.dense_run(1969060699)
    tf.assert_equal(computed, (0.7312801, -0.6338747))

  def test_bottleneck_block_width_4_channels_4_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        1716369119, batch_size=32, bottleneck=True, projection=True,
        width=4, channels=4)
    tf.assert_equal(computed_size, (32, 2, 2, 8))
    tf.assert_equal(computed_values, (0.41549513, 0.27814695, -190.2442))

  def test_bottleneck_block_width_4_channels_8_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        1455996458, batch_size=32, bottleneck=True, projection=True,
        width=4, channels=8)
    tf.assert_equal(computed_size, (32, 2, 2, 16))
    tf.assert_equal(computed_values, (1.1036423, 1.1127403, 143.32068))

  def test_bottleneck_block_width_4_channels_32_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        2770738568, batch_size=32, bottleneck=True, projection=True,
        width=4, channels=32)
    tf.assert_equal(computed_size, (32, 2, 2, 64))
    tf.assert_equal(computed_values, (-0.46288937, -1.0053508, 147.64088))

  def test_bottleneck_block_width_32_channels_4_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        1262621774, batch_size=32, bottleneck=True, projection=True,
        width=32, channels=4)
    tf.assert_equal(computed_size, (32, 16, 16, 8))
    tf.assert_equal(computed_values, (-0.36800718, -0.41594106, 9638.111))

  def test_bottleneck_block_width_32_channels_8_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        3856195393, batch_size=32, bottleneck=True, projection=True,
        width=32, channels=8)
    tf.assert_equal(computed_size, (32, 16, 16, 16))
    tf.assert_equal(computed_values, (1.0703044, 0.7402196, 9908.163))

  def test_bottleneck_block_width_32_channels_32_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        2246470031, batch_size=32, bottleneck=True, projection=True,
        width=32, channels=32)
    tf.assert_equal(computed_size, (32, 16, 16, 64))
    tf.assert_equal(computed_values, (-0.17540368, 0.25853187, 26784.121))

  def test_bottleneck_block_width_128_channels_4_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        2718968685, batch_size=32, bottleneck=True, projection=True,
        width=128, channels=4)
    tf.assert_equal(computed_size, (32, 64, 64, 8))
    tf.assert_equal(computed_values, (-1.3875123, 0.46859837, 240767.08))

  def test_bottleneck_block_width_128_channels_8_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        1976723312, batch_size=32, bottleneck=True, projection=True,
        width=128, channels=8)
    tf.assert_equal(computed_size, (32, 64, 64, 16))
    tf.assert_equal(computed_values, (-0.33272713, 0.9874536, 29585.953))

  def test_bottleneck_block_width_128_channels_32_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        3519054312, batch_size=32, bottleneck=True, projection=True,
        width=128, channels=32)
    tf.assert_equal(computed_size, (32, 64, 64, 64))
    tf.assert_equal(computed_values, (-0.11728054, 0.3465855, 81544.01))

  def test_bottleneck_block_width_4_channels_4_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        2450274501, batch_size=32, bottleneck=True, projection=False,
        width=4, channels=4)
    tf.assert_equal(computed_size, (32, 4, 4, 4))
    tf.assert_equal(computed_values, (0.6107404, 0.060905337, 981.8533))

  def test_bottleneck_block_width_4_channels_8_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        158822092, batch_size=32, bottleneck=True, projection=False,
        width=4, channels=8)
    tf.assert_equal(computed_size, (32, 4, 4, 8))
    tf.assert_equal(computed_values, (-0.20900649, 0.9496709, 1630.238))

  def test_bottleneck_block_width_4_channels_32_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        1827880642, batch_size=32, bottleneck=True, projection=False,
        width=4, channels=32)
    tf.assert_equal(computed_size, (32, 4, 4, 32))
    tf.assert_equal(computed_values, (0.31675977, 0.6780378, 7017.4956))

  def test_bottleneck_block_width_32_channels_4_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        4281570515, batch_size=32, bottleneck=True, projection=False,
        width=32, channels=4)
    tf.assert_equal(computed_size, (32, 32, 32, 4))
    tf.assert_equal(computed_values, (0.65464973, 0.9816817, 73392.24))

  def test_bottleneck_block_width_32_channels_8_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        2498606963, batch_size=32, bottleneck=True, projection=False,
        width=32, channels=8)
    tf.assert_equal(computed_size, (32, 32, 32, 8))
    tf.assert_equal(computed_values, (0.26228523, 1.3094232, 130402.22))

  def test_bottleneck_block_width_32_channels_32_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        302237443, batch_size=32, bottleneck=True, projection=False,
        width=32, channels=32)
    tf.assert_equal(computed_size, (32, 32, 32, 32))
    tf.assert_equal(computed_values, (0.54678255, -0.39149415, 334033.4))

  def test_bottleneck_block_width_128_channels_4_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        1163736720, batch_size=32, bottleneck=True, projection=False,
        width=128, channels=4)
    tf.assert_equal(computed_size, (32, 128, 128, 4))
    tf.assert_equal(computed_values, (0.77162766, 0.6421704, 1085749.9))

  def test_bottleneck_block_width_128_channels_8_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        951433898, batch_size=32, bottleneck=True, projection=False,
        width=128, channels=8)
    tf.assert_equal(computed_size, (32, 128, 128, 8))
    tf.assert_equal(computed_values, (0.2405963, 0.9551655, 2548026.8))

  def test_bottleneck_block_width_128_channels_32_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        1196661117, batch_size=32, bottleneck=True, projection=False,
        width=128, channels=32)
    tf.assert_equal(computed_size, (32, 128, 128, 32))
    tf.assert_equal(computed_values, (0.25506544, 0.17008033, 9960076.0))

  def test_building_block_width_4_channels_4_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        199438964, batch_size=32, bottleneck=False, projection=True,
        width=4, channels=4)
    tf.assert_equal(computed_size, (32, 2, 2, 8))
    tf.assert_equal(computed_values, (0.71276045, 0.08712906, 24.600151))

  def test_building_block_width_4_channels_8_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        1846153380, batch_size=32, bottleneck=False, projection=True,
        width=4, channels=8)
    tf.assert_equal(computed_size, (32, 2, 2, 16))
    tf.assert_equal(computed_values, (0.2906993, 1.4324092, 45.3777))

  def test_building_block_width_4_channels_32_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        2157012381, batch_size=32, bottleneck=False, projection=True,
        width=4, channels=32)
    tf.assert_equal(computed_size, (32, 2, 2, 64))
    tf.assert_equal(computed_values, (0.10871497, -0.03330171, 634.4708))

  def test_building_block_width_32_channels_4_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        1601363320, batch_size=32, bottleneck=False, projection=True,
        width=32, channels=4)
    tf.assert_equal(computed_size, (32, 16, 16, 8))
    tf.assert_equal(computed_values, (-1.453328, 0.7330329, -7662.508))

  def test_building_block_width_32_channels_8_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        2408074324, batch_size=32, bottleneck=False, projection=True,
        width=32, channels=8)
    tf.assert_equal(computed_size, (32, 16, 16, 16))
    tf.assert_equal(computed_values, (0.69784254, 0.3302448, 10008.539))

  def test_building_block_width_32_channels_32_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        966519200, batch_size=32, bottleneck=False, projection=True,
        width=32, channels=32)
    tf.assert_equal(computed_size, (32, 16, 16, 64))
    tf.assert_equal(computed_values, (0.014134437, -0.35302147, 2308.9548))

  def test_building_block_width_128_channels_4_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        2799845638, batch_size=32, bottleneck=False, projection=True,
        width=128, channels=4)
    tf.assert_equal(computed_size, (32, 64, 64, 8))
    tf.assert_equal(computed_values, (1.2411811, 0.36829096, -31472.748))

  def test_building_block_width_128_channels_8_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        303401096, batch_size=32, bottleneck=False, projection=True,
        width=128, channels=8)
    tf.assert_equal(computed_size, (32, 64, 64, 16))
    tf.assert_equal(computed_values, (1.1224514, -0.3613649, 208603.86))

  def test_building_block_width_128_channels_32_batch_size_32_with_proj(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        3857876037, batch_size=32, bottleneck=False, projection=True,
        width=128, channels=32)
    tf.assert_equal(computed_size, (32, 64, 64, 64))
    tf.assert_equal(computed_values, (0.625062, 0.06916532, -893260.1))

  def test_building_block_width_4_channels_4_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        3948561567, batch_size=32, bottleneck=False, projection=False,
        width=4, channels=4)
    tf.assert_equal(computed_size, (32, 4, 4, 4))
    tf.assert_equal(computed_values, (1.3410637, -0.010343775, 846.4164))

  def test_building_block_width_4_channels_8_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        3944950661, batch_size=32, bottleneck=False, projection=False,
        width=4, channels=8)
    tf.assert_equal(computed_size, (32, 4, 4, 8))
    tf.assert_equal(computed_values, (0.65315914, 0.40512276, 2367.1187))

  def test_building_block_width_4_channels_32_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        2851841486, batch_size=32, bottleneck=False, projection=False,
        width=4, channels=32)
    tf.assert_equal(computed_size, (32, 4, 4, 32))
    tf.assert_equal(computed_values, (-0.39765465, 0.8226229, 8434.205))

  def test_building_block_width_32_channels_4_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        4091441499, batch_size=32, bottleneck=False, projection=False,
        width=32, channels=4)
    tf.assert_equal(computed_size, (32, 32, 32, 4))
    tf.assert_equal(computed_values, (1.7269518, 1.4094924, 42441.63))

  def test_building_block_width_32_channels_8_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        1884045710, batch_size=32, bottleneck=False, projection=False,
        width=32, channels=8)
    tf.assert_equal(computed_size, (32, 32, 32, 8))
    tf.assert_equal(computed_values, (1.1263828, 0.2826241, 172352.58))

  def test_building_block_width_32_channels_32_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        934342683, batch_size=32, bottleneck=False, projection=False,
        width=32, channels=32)
    tf.assert_equal(computed_size, (32, 32, 32, 32))
    tf.assert_equal(computed_values, (0.96356857, 0.719787, 531449.1))

  def test_building_block_width_128_channels_4_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        3054946484, batch_size=32, bottleneck=False, projection=False,
        width=128, channels=4)
    tf.assert_equal(computed_size, (32, 128, 128, 4))
    tf.assert_equal(computed_values, (0.9403712, 0.8296507, 630401.3))

  def test_building_block_width_128_channels_8_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        3970466773, batch_size=32, bottleneck=False, projection=False,
        width=128, channels=8)
    tf.assert_equal(computed_size, (32, 128, 128, 8))
    tf.assert_equal(computed_values, (0.61943376, 0.7390875, 2140461.5))

  def test_building_block_width_128_channels_32_batch_size_32(self):
    """Test of a single ResNet block."""
    computed_size, computed_values = self.resnet_block_run(
        2953171569, batch_size=32, bottleneck=False, projection=False,
        width=128, channels=32)
    tf.assert_equal(computed_size, (32, 128, 128, 32))
    tf.assert_equal(computed_values, (0.32318482, 0.8760831, 10266849.0))


class TestParser(argparse.ArgumentParser):

  def __init__(self):
    super(TestParser, self).__init__()
    self.add_argument("--remake_tests", action="store_true",
                      help="Regenerate test cases rather than running them.")


if __name__ == "__main__":
  parser = TestParser()
  flags = parser.parse_args()
  if flags.remake_tests:
    if sys.version_info[0] == 2:
      raise SystemError(
          "Python 2 does not support the instantiation of unittest.TestCase "
          "objects directly. If you wish to regenerate the tests please use "
          "python 3.")
    block_tests = BlockTest()
    block_tests.generate_tests()
  else:
    tf.test.main()
