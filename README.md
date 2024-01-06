# cumesh2sdf
A mesh to SDF algorithm implemented with CUDA.

## PyTorch Plugin

### Installation

You will need a working version of PyTorch and CUDA development toolkit.
And install [`diso`](https://github.com/SarahWeiii/diso) python package.

```bash
pip install diso
pip install git+https://github.com/eliphatfs/cumesh2sdf.git
```

Import the library as:
```python
import torch
import torchcumesh2sdf
```

### API

#### `get_sdf`

> ```python
> torchcumesh2sdf.get_sdf(
>     tris: torch.Tensor,
>     R: int, band: float,
>     B: int = 16384
> ) -> torch.Tensor
> ```

**Parameters:**
+ `tris`: input triangle soup of shape `(F, 3, 3)` with dtype `float32`.
+ `R`: resolution of output SDF volume. **Required to be power of 2.**
+ `band`: size of the band near the mesh to compute values. In regions farther than the band, only the sign with a large value `1e9` will be set. Recommended values: `3/R` or `4/R`.
+ `B`: batch size of triangles fed at once to the algorithm. This has no effect on the results (but can affect performance), unless the algorithm meets overflow of `int32` indices. It is recommended to leave as default unless you see a warning message.

**Returns:**
`(R, R, R)` SDF volume with dtype `float32`.

#### `get_udf`

> ```python
> torchcumesh2sdf.get_udf(
>     tris: torch.Tensor,
>     R: int, band: float,
>     B: int = 16384
> ) -> torch.Tensor
> ```

Parameters are the same as `get_sdf`, but this function instructs the algorithm to generate an **unsigned** distance field volume. This is faster than `get_sdf`.

#### `free_cached_memory`

> ```python
> torchcumesh2sdf.free_cached_memory()
> ```

`torchcumesh2sdf` caches allocations to make computation faster. You may call this method to free any cached memory allocations.

### Example

See [example.py](example.py) for the mesh2volume example script.

See [test.py](test.py) for the GPU remesh example script.

See [test_openvdb.py](test_openvdb.py) for the OpenVDB remesh example script.

## Standalone

Clone and compile with:

```bash
bash build.sh
```
