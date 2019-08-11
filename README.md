# Einig: Tensorflow einsum generalizations

This package provides wrappers around `tf.einsum` to add support for:

  * Ellipsis `...` to match any number of axis, e.g. `einsum("...ij,...jk->...ik", x, y)`
    for a `batched_matmul` that works with any number of leading batch axes.
  * Named composites: capital letters (by default) are treated the same as `...`, so now
    you can have multiple composites: `einsum("Ij,jK->IK", x, y)` for a `matmul` between
    higher-order linear maps. Note that unfortunately, something like `IJ,JK->IK`
    wouldn't work, as it may in general be ambiguous.
  * `keepdims`/`newaxis` using the underscore character, e.g. `einsum("ij->i_", x)` is
    equivalent to `tf.reduce_sum(x, axis=1, keepdims=True)`.
  * Extracting diagonal parts: `ii->i`, `bii->bi` as supported by `tf.linalg.diag_part`,
    but also `bijkijlj,bklji->bijk` or what have you.
  * Inducing diagonal axes: `i->ii`, which somehow isn't supported by `np.einsum` either.

All of these are supported simultaneously by `einig.einsum`.

