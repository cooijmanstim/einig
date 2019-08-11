import string
import itertools, functools
import tensorflow as tf, numpy as np
import util
from collections import namedtuple

def einsum(sig, *xs, _einsum=tf.einsum, **kwargs):
  sig = Signature.make(sig)
  assert sig.arity == len(xs)
  xs = tuple(map(tf.convert_to_tensor, xs))
  @EinsumFeatures.composites
  @EinsumFeatures.newaxis
  @EinsumFeatures.diag_in
  @EinsumFeatures.diag_out
  def wrapped_einsum(sig, xs, **kwargs):
    return _einsum(sig.stringify(), *xs, **kwargs)
  return wrapped_einsum(sig, xs, **kwargs)

def eincount(signature, *args, _einsum=einsum, **einsum_kwargs):
  # use this function to count (i.e. sum) across binary arrays.
  # (it is here because np.einsum does `einany` on bools, which can be unexpected.)
  assert all(map(is_bool, args))
  args = tuple(tf.cast(arg, tf.int32) for arg in args)
  return einsum(signature, *args, **einsum_kwargs)

def einany(signature, *args, _einsum=einsum, **einsum_kwargs):
  # use this function to reduce_or across binary arrays.
  assert all(map(is_bool, args))
  # cast to int to make tf.einsum work
  args = tuple(tf.cast(arg, tf.int32) for arg in args)
  result = einsum(signature, *args, **einsum_kwargs)
  # cast back to bool: OR <=> at least one
  result = result >= 1
  assert is_bool(result)
  return result

class EinsumFeatures:
  def _make_einsum_wrapper(fn):
    def einsum_wrapper(inner_fn):
      return functools.partial(fn, inner_fn)
    return einsum_wrapper

  @_make_einsum_wrapper
  def composites(_yield, sig, xs, **kwargs):
    assert isinstance(sig, Signature)
    sig = expand_composites(sig, xs)
    return _yield(sig, xs, **kwargs)
  @_make_einsum_wrapper
  def newaxis(_yield, sig, xs, **kwargs):
    assert isinstance(sig, Signature)
    assert not any("_" in xsig for xsig in sig.xs)
    y = _yield(sig.replace("_", ""), xs, **kwargs)
    idx = tuple(None if c == "_" else slice(None) for c in sig.y)
    return y[idx]
  @_make_einsum_wrapper
  def diag_in(_yield, sig, xs, **kwargs):
    assert isinstance(sig, Signature)
    sig, xs = resolve_diag_input(sig, xs)
    return _yield(sig, xs, **kwargs)
  @_make_einsum_wrapper
  def diag_out(_yield, sig, xs, **kwargs):
    assert isinstance(sig, Signature)
    y = resolve_diag_output(sig, lambda sig: _yield(sig, xs, **kwargs))
    return y

class Signature(namedtuple("Signature", "xs y")):
  # internally we map "..." to a single character so that we can process expressions
  # character-by-character rather than having to build a parser and so on.
  Ellipsis = "*"

  @classmethod
  def parse(cls, s):
    # replace ellipses by a single-character symbol so we don't need to do any parsing
    assert cls.Ellipsis not in s
    s = s.replace("...", cls.Ellipsis)

    x, y = s.split("->")
    xs = x.split(",")

    # some basic sanity checks
    valid_characters = (cls.Ellipsis, "_", *string.ascii_letters)
    assert all(c in valid_characters for c in y)
    for x in xs:
      assert all(c in valid_characters for c in x)

    return cls(tuple(xs), y)

  arity = property(lambda self: len(self.xs))

  def stringify(self):
    return "->".join((",".join(self.xs), self.y)).replace(self.Ellipsis, "...")

  def replace(self, a, b):
    return Signature(xs=tuple(x.replace(a, b) for x in self.xs),
                     y=self.y.replace(a, b))

  def as_grouping(self):
    """Convert this Signature to a `grouping` structure for use with `diag_scatter`/`diag_gather`.

    Only unary, non-reducing signatures can be converted to such groupings."""
    if len(self.xs) != 1:
      # there *is* a natural generalization of the `grouping` structure to
      # multiple arguments: each input would have its own grouping, with as
      # many groups as there are output axes.
      raise ValueError("only unary Signatures can be converted to groupings")
    x, = self.xs
    _, grouping = _grouping(x)
    return grouping

Signature.make = functools.singledispatch(util.notimplemented)
Signature.make.register(Signature)(lambda sig: Signature(*sig))
Signature.make.register(str)(lambda s: Signature.parse(s))

def diag_gather(x, grouping):
  """Extract diagonal subtensors from `x`.

  This is a generalization of `tf.linalg.diag_part` to multiple dimensions and
  multiple diagonals.

  `grouping` is a list of lists, each of which contains axes whose diagonal
  should be taken. Order of groups determines order of axes in the output.
  Groups may contain any nonzero number of axes for simultaneous
  diagonalization across multiple axes, and axes within groups need not be
  contiguous. The grouping must form a partition: all axes of `x` must appear
  exactly once. A `grouping` may be obtained from an einsum signature through
  `Signature.as_grouping`, e.g. `Signature("ii->i").as_grouping()`.

  Examples:
    `diag_gather(x, [[0, 1]]) <=> einsum("ii->i", x)`
    `diag_gather(x, [[0], [1, 2]]) <=> einsum("bii->bi", x)`
  """
  x_shape = get_shape(x)

  if not all(grouping):
    # TODO support this if a use case appears
    raise ValueError("empty groups are not allowed", grouping)
  shapes = [[x_shape[axis] for axis in group] for group in grouping]
  if not all(util.all_equal(shape) for shape in shapes):
    raise ValueError("grouped axes must have equal dimension", grouping, shapes)
  if sorted(axis for group in grouping for axis in group) != list(range(len(x_shape))):
    # TODO consider supporting Ellipsis if it can be made predictable
    raise ValueError("grouping must reference all axes of x exactly once", grouping)

  strides = [sum(x_shape[axes[0]] ** k for k in range(len(axes)))
             for axes in grouping]
  # transpose the argument so that the tied axes are adjacent
  z = tf.transpose(x, [axis for axes in grouping for axis in axes])
  # flatten the tied axes and use strided slicing to select the diagonal elements
  z = tf.reshape(z, [util.prod([x_shape[axis] for axis in axes])
                     for axes in grouping])
  y = z[tuple(slice(None, None, stride)
              for stride in strides)]
  return y

def diag_scatter(x, grouping):
  """Insert diagonal axes into `x`.

  This is a generalization of `tf.linalg.diag`, and the transpose and/or inverse of
  `diag_gather`.

  `grouping` is a list of lists similar to that for `diag_gather` in the sense
  that if `y == diag_scatter(x, grouping)`, then `x == diag_gather(y, grouping)`
  up to a loss of off-diagonal elements.

  Examples:
    `diag_scatter(x, [[0, 1]]) <=> einsum("i->ii", x)`
    `diag_scatter(x, [[0], [1, 2]]) <=> einsum("bi->bii", x)`
  """
  x_shape = get_shape(x)

  if not len(grouping) == len(x_shape):
    raise ValueError("grouping must contain one group for each axis of x", grouping)
  if not all(grouping):
    raise ValueError("empty groups are not allowed", grouping)
  flat_grouping = [axis for axes in grouping for axis in axes]
  if sorted(flat_grouping) != list(range(len(flat_grouping))):
    # TODO be less confusing? esp in the case of gaps
    raise ValueError("grouping axes must reference all axes of the result exactly once")

  # this stuff is hairy. simple example case in 2d:
  # [a b c]
  # pad to [0 0 0 0 0 0 0 0 0 a b c]
  # reshape to [[0 0 0] [0 0 0] [0 0 0] [a b c]]
  # transpose [[0 0 0 a] [0 0 0 b] [0 0 0 c]]
  # reshape to [0 0 0 a 0 0 0 b 0 0 0 c]
  # cut off excess: [a 0 0 0 b 0 0 0 c]
  # reshape to [[a 0 0] [0 b 0] [0 0 c]]
  # there's more direct ways using concat/stack and reshape, but this way we
  # can operate on all axes/groupings simultaneously.

  strides = [sum(x_shape[x_axis] ** k for k in range(len(y_axes)))
             for x_axis, y_axes in enumerate(grouping)]

  # blow up the dimensions, padding each with zeros in front
  z = tf.pad(x, [((stride - 1) * x_shape[x_axis], 0)
                 for x_axis, stride in enumerate(strides)])
  # reshape each axis of x to be 2d
  z = tf.reshape(z, [dim
                     for x_axis, stride in enumerate(strides)
                     for dim in [stride, x_shape[x_axis]]])
  # transpose the new pairs of axes
  z = tf.transpose(z, [axis
                       for x_axis in range(len(x_shape))
                       for axis in (2 * x_axis + 1, 2 * x_axis)])
  # flatten them again
  z = tf.reshape(z, [stride * x_shape[x_axis]
                     for x_axis, stride in enumerate(strides)])
  # cut off excess zeros from the front
  assert all(stride >= 1 for stride in strides) # avoid negative indexing bugs
  z = z[tuple(slice(stride - 1, None) for stride in strides)]
  # reshape again into their final shape
  z = tf.reshape(z, [x_shape[x_axis]
                     for x_axis, y_axes in enumerate(grouping)
                     for _ in y_axes])
  # transpose to move the related axes into the positions specified in grouping
  y_axes_by_z_axes = [axis for y_axes in grouping for axis in y_axes]
  z_axes_by_y_axes = [y_axes_by_z_axes.index(z_axis) for z_axis in range(len(y_axes_by_z_axes))]
  y = tf.transpose(z, z_axes_by_y_axes)
  return y


def expand_composites(sig, xs, symbols=(Signature.Ellipsis, *string.ascii_uppercase)):
  """Expand composites in Signature `sig` based on ranks of arguments `xs`.

  `np.einsum` supports a special placeholder "..." (ellipsis) that will match
  any number of contiguous dimensions. This function handles the expansion of a
  generalized notion of ellipsis. In addition to `np.einsum`'s one-off "...",
  this function supports any-ish number of *named* composites, which by default
  are indicated by ascii uppercase letters. Note that this behavior is
  incompatible with `np.einsum`, which treats ascii uppercase letters as regular
  single-axis symbols.

  Examples:
    `einsum("I,I->", x, y)` for a generalized scalar product.
    `einsum("IJ,JK->IK", x, y)` for a generalized inner product.
    `einsum("I,J->IJ", x, y)` for a generalized outer product.
    `einsum("Bij,Bjk->Bik", x, y)` for batched inner product that's agnostic about
      the number of batch axes B.
  """
  composites = set(c for c in sig.stringify() if c in symbols)
  if not composites:
    return sig

  # determine shapes underneath composite axes
  unresolved = set(composites)
  ndims = dict()
  while unresolved:
    # we resolve the composites by repeatedly going through the input
    # signatures and finding one that has only one as yet unresolved
    # composite. there may be cases in which this fails to figure it out even
    # if the overall einsum signature is technically unambiguous.
    for xsig, x in zip(sig.xs, xs):
      candidates = set(xsig) & unresolved
      if len(candidates) == 1:
        composite, = candidates
        explained_ndim = sum(ndims.get(c, 0)
                            if c in symbols else 1
                            for c in xsig)
        unresolved_ndim = get_ndim(x) - explained_ndim
        # if the symbol appears multiple times, distribute evenly.
        multiplicity = xsig.count(composite)
        assert unresolved_ndim % multiplicity == 0 # TODO raise ValueError
        ndims[composite] = unresolved_ndim // multiplicity
        unresolved.remove(composite)
        break
    else:
      # stop if no progress in the last iteration
      break
  if unresolved:
    raise ValueError("could not resolve composite ranks in expression", sig.stringify(), xs)

  # the composites are implemented under the hood by expanding them into
  # sequences of regular lowercase symbols.
  unused_alphabet = "".join(sorted(set(string.ascii_lowercase) - set(sig.stringify())))
  expansions = dict()
  for composite, ndim in ndims.items():
    if len(unused_alphabet) < ndim:
      # NOTE at this point it's technically possible to expand our range to
      # uppercase index symbols if ever needed
      raise ValueError("not enough indices to expand composites", sig.stringify(), xs)
    expansions[composite], unused_alphabet = unused_alphabet[:ndim], unused_alphabet[ndim:]

  for composite, expansion in expansions.items():
    sig = sig.replace(composite, expansion)
  return sig

def resolve_diag_input(sig, xs):
  """Handle ii->i for extracting diagonals."""
  # at this point, characters ought to correspond to singular axes
  assert all(len(xsig) == get_ndim(x) for xsig, x in util.eqzip(sig.xs, xs))

  def process_x(xsig, x):
    newxsig, grouping = _grouping(xsig)
    if newxsig == xsig:
      # no repeated input indices; nothing to be done
      return xsig, x
    newx = diag_gather(x, grouping)
    return newxsig, newx
  xsigs, xs = util.eqzip(*itertools.starmap(process_x, util.eqzip(sig.xs, xs)))
  return Signature(xs=xsigs, y=sig.y), xs

def resolve_diag_output(sig, _callback):
  """Handle i->ii for inducing diagonals.

  This needs to wrap around an inner einsum call because it needs to both preprocess
  the output signature and post-process the output."""
  ysig, grouping = _grouping(sig.y)
  if ysig == sig.y:
    # no repeated output indices; nothing to be done
    return _callback(sig)
  y = _callback(Signature(xs=sig.xs, y=ysig))
  newy = diag_scatter(y, grouping)
  return newy


def _grouping(s):
  repeats = util.ordict()
  for i, c in enumerate(s):
    repeats.setdefault(c, []).append(i)
  return "".join(repeats.keys()), list(repeats.values())

def is_bool(x):
  dtype = x.dtype
  try:
    # if x is a tensorflow object, check its numpy dtype
    dtype = dtype.as_numpy_dtype
  except AttributeError:
    pass
  return np.issubdtype(dtype, np.bool_)

def get_shape(x):
  shape = x.shape
  if isinstance(shape, tf.TensorShape):
    shape = shape.as_list()
  shape = list(shape)
  if None in shape:
    # sorry
    raise ValueError("shape must be fully defined")
  return shape

def get_ndim(x):
  try:
    return x.ndim
  except AttributeError:
    return x.shape.rank
