# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests feeding functions using arrays and `DataFrames`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np

from tensorflow.python.estimator.inputs.queues import feeding_functions as ff
from tensorflow.python.platform import test


def vals_to_list(a):
  return {
      key: val.tolist() if isinstance(val, np.ndarray) else val
      for key, val in a.items()
  }


class _FeedingFunctionsTestCase(test.TestCase):
  """Tests for feeding functions."""

  def testOrderedDictNumpyFeedFnBatchTwoWithOneEpoch(self):
    a = np.arange(32, 42).reshape([5, 2])
    b = np.arange(64, 69)
    x = {"a": a, "b": b}
    ordered_dict_x = collections.OrderedDict(
        sorted(x.items(), key=lambda t: t[0]))
    placeholders = ["index_placeholder", "a_placeholder", "b_placeholder"]
    aff = ff._OrderedDictNumpyFeedFn(
        placeholders, ordered_dict_x, batch_size=2, num_epochs=1)

    expected = {
        "index_placeholder": [0, 1],
        "a_placeholder": [[32, 33], [34, 35]],
        "b_placeholder": [64, 65]
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))

    expected = {
        "index_placeholder": [2, 3],
        "a_placeholder": [[36, 37], [38, 39]],
        "b_placeholder": [66, 67]
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))

    expected = {
        "index_placeholder": [4],
        "a_placeholder": [[40, 41]],
        "b_placeholder": [68]
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))

  def testOrderedDictNumpyFeedFnLargeBatchWithSmallArrayAndMultipleEpochs(self):
    a = np.arange(32, 36).reshape([2, 2])
    b = np.arange(64, 66)
    x = {"a": a, "b": b}
    ordered_dict_x = collections.OrderedDict(
        sorted(x.items(), key=lambda t: t[0]))
    placeholders = ["index_placeholder", "a_placeholder", "b_placeholder"]
    aff = ff._OrderedDictNumpyFeedFn(
        placeholders, ordered_dict_x, batch_size=100, num_epochs=2)

    expected = {
        "index_placeholder": [0, 1, 0, 1],
        "a_placeholder": [[32, 33], [34, 35], [32, 33], [34, 35]],
        "b_placeholder": [64, 65, 64, 65]
    }
    actual = aff()
    self.assertEqual(expected, vals_to_list(actual))


if __name__ == "__main__":
  test.main()
