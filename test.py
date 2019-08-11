import itertools
import tensorflow as tf, numpy as np
import einig, util

class DiagTest(tf.test.TestCase):
  def test_diag_gather(self):
    with self.cached_session() as sess:
      x = _make_test_array([3, 3])
      self.assertAllEqual(np.einsum("ii->i", x),
                          einig.diag_gather(x, [[0, 1]]).eval())
      x = _make_test_array([2, 3, 5, 3, 2])
      self.assertAllEqual(np.einsum("ijkji->ikj", x),
                          einig.diag_gather(x, [[0, 4], [2], [1, 3]]).eval())

  def test_diag_scatter(self):
    with self.cached_session() as sess:
      x = _make_test_array([3])
      self.assertAllEqual(np.diag(x),
                          einig.diag_scatter(x, [[0, 1]]).eval())
      x = _make_test_array([2, 3, 5])
      self.assertAllEqual(np.einsum("ijk,jl,im->ijklm", x, np.eye(3), np.eye(2)),
                          einig.diag_scatter(x, [[0, 4], [1, 3], [2]]).eval())

  def test_diag_consistent(self):
    with self.cached_session() as sess:
      x = _make_test_array([5])
      for grouping in [
          [[0, 1]],
          [[0, 1, 2]],
          [[0, 1, 2, 3]],
      ]:
        x_scatter = einig.diag_scatter(x, grouping).eval()
        x_gather = einig.diag_gather(x_scatter, grouping).eval()
        x_scatter2 = einig.diag_scatter(x_gather, grouping).eval()
        self.assertAllEqual(x, x_gather)
        self.assertAllEqual(x_scatter, x_scatter2)

      x = _make_test_array([2, 3, 5])
      for grouping in [
          [[0, 3], [2, 1, 5], [4]],
          [[0, 4], [3, 2, 5], [1]],
      ]:
        x_scatter = einig.diag_scatter(x, grouping).eval()
        x_gather = einig.diag_gather(x_scatter, grouping).eval()
        x_scatter2 = einig.diag_scatter(x_gather, grouping).eval()
        self.assertAllEqual(x, x_gather)
        self.assertAllEqual(x_scatter, x_scatter2)


class EinsumTest(tf.test.TestCase):
  def assert_einsums_equal(self, signature, x):
    with self.cached_session() as sess:
      self.assertAllEqual(np.einsum(signature, x), einig.einsum(signature, x).eval())

  def test_blockdiag_input(self):
    self.assert_einsums_equal("ii->i", _make_test_array([3, 3]))
    self.assert_einsums_equal("ijklji->ikjl", _make_test_array([2, 3, 5, 6, 3, 2]))

  def test_blockdiag_output(self):
    x = _make_test_array([3])
    self.assertAllEqual(np.diag(x), einig.einsum("i->ii", x))
    x = _make_test_array([3, 5])
    y = np.zeros([5, 3, 5, 3, 5])
    for i, j in itertools.product(range(3), range(5)):
      y[j, i, j, i, j] = x[i, j]
    self.assertAllEqual(y, einig.einsum("ij->jijij", x))

  def test_named_composites(self):
    x = _make_test_array([2, 3])
    y = _make_test_array([2, 3])
    z = _make_test_array([2, 3, 5])
    self.assertAllEqual(np.einsum("ij,ij,ijk->k", x, y, z),
                        einig.einsum("BI,I,IJ->BJ", x, y, z))

  def test_expand_dims(self):
    x = _make_test_array([2, 3])
    y = _make_test_array([3, 5])
    self.assertAllEqual(np.einsum("ij,jk->ik", x, y)[None, :, None, :, None],
                        einig.einsum("ij,jk->_i_k_", x, y))

def _make_test_array(shape):
  return (1 + np.arange(util.prod(shape))).reshape(shape)

