import functools, operator
from collections import OrderedDict as ordict, defaultdict as ddict

def all_equal(xs, key=None):
  if key is not None:
    xs = list(map(key, xs))
  return all(x == xs[0] for x in xs[1:])

def eqzip(*xss):
  xss = list(map(list, xss))
  assert all_equal(xss, key=len)
  return zip(*xss)

def prod(xs):
  return functools.reduce(operator.mul, xs, 1)

def notimplemented(*args, **kwargs):
  raise NotImplementedError()

def ifnone(x, y):
  return y() if x is None else x

