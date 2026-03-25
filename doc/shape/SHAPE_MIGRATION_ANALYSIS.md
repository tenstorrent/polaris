# Shape Class Migration Analysis

## Summary

The `ttsim.ops.tensor.Shape` class has been changed from subclassing `list` to being a container class with a `_shape` member. This document identifies all code locations that need to be updated to work with the new implementation.

## Key Changes Required

1. **Direct access to `._shape`**: Replace with public API (`shape.view()`, `list(shape)`, or iteration)
2. **List concatenation with Shape**: Convert Shape to list before concatenation
3. **`.copy()` method**: Shape no longer has `.copy()`, use `Shape(shape)` or `list(shape)`
4. **Slicing and concatenation**: Ensure proper conversion when mixing Shape with lists

## Files Requiring Changes

### 1. `ttsim/front/ttnn/tensor.py`

**Issues:**
- Line 237: Direct access to `logical_shape._shape` - should use `list(logical_shape)` or `logical_shape.view()`
- Line 266: List concatenation `self.shape[:dim] + [1] + self.shape[dim:]` - convert shape to list first
- Line 277: List concatenation `self.shape[:dim] + self.shape[dim+1:]` - convert shape to list first
- Line 299: Comparison `self.shape == [1]` - should work due to `__eq__`, but verify
- Line 308: Direct access to `self.shape._shape` - should use `self.shape.view()` or `list(self.shape)`
- Line 326: Direct access to `original_shape._shape` - should use `list(original_shape)` or `original_shape.view()`
- Line 328: Direct access to `original_shape._shape` - should use `list(original_shape)` or `original_shape.view()`

**Required Changes:**
```python
# Line 237: Change
padded = list(logical_shape._shape)
# To:
padded = list(logical_shape)  # or logical_shape.view()

# Line 266: Change
new_shape = self.shape[:dim] + [1] + self.shape[dim:]
# To:
new_shape = list(self.shape[:dim]) + [1] + list(self.shape[dim:])
# Or:
shape_list = list(self.shape)
new_shape = shape_list[:dim] + [1] + shape_list[dim:]

# Line 277: Change
new_shape = self.shape[:dim] + self.shape[dim+1:]
# To:
shape_list = list(self.shape)
new_shape = shape_list[:dim] + shape_list[dim+1:]

# Line 308: Change
return tuple(self.shape._shape)
# To:
return tuple(self.shape)  # or tuple(self.shape.view())

# Lines 326, 328: Change
padded_original = [1] * (len(target_shape) - len(original_shape)) + original_shape._shape
padded_original = original_shape._shape
# To:
padded_original = [1] * (len(target_shape) - len(original_shape)) + list(original_shape)
padded_original = list(original_shape)
```

### 2. `ttsim/front/ttnn/ttnn_shim.py`

**Issues:**
- Line 851: Direct access to `shape._shape` - should use `list(shape)` or `shape.view()`
- Line 898: Direct access to `shape._shape` - should use `list(shape)` or `shape.view()`
- Line 908: Direct access to `tensor.logical_shape()._shape` and `shape._shape` - should use public API
- Line 933: Direct access to `shape._shape` - should use `list(shape)` or `shape.view()`
- Line 960: Direct access to `padded_shape._shape` - should use `list(padded_shape)` or `padded_shape.view()`
- Line 1051: Direct access to `output_shape._shape` - should use `list(output_shape)` or `output_shape.view()`
- Line 1081: Direct access to `output_shape._shape` - should use `list(output_shape)` or `output_shape.view()`
- Line 1229: Direct access to `output_shape._shape` - should use `list(output_shape)` or `output_shape.view()`
- Line 1359: Direct access to `output_padded_shape._shape` - should use public API
- Line 1367: Direct access to `output_padded_shape._shape` - should use public API
- Line 1471: Direct access to `padded_shape._shape` and `output_shape._shape` - should use public API
- Line 1472: Direct access to `output_shape._shape` - should use public API
- Line 1479: Direct access to `output_shape._shape` - should use public API
- Line 1587: Direct access to `output_shape._shape` - should use `list(output_shape)` or `output_shape.view()`
- Line 1607: Direct access to `logical_shape._shape` - should use `list(logical_shape)` or `logical_shape.view()`
- Line 1679: Direct access to `logical_shape._shape` - should use `list(logical_shape)` or `logical_shape.view()`
- Line 1722: Direct access to `padded_shape._shape` - should use `list(padded_shape)` or `padded_shape.view()`

**Required Changes:**
All instances of `shape._shape` should be replaced with `list(shape)` or `shape.view()`.

### 3. `ttsim/ops/desc/tensor.py`

**Issues:**
- Line 103: List comprehension over `X.shape` - should work due to `__iter__`, but verify
- Line 131: List comprehension over `dataT.shape` - should work due to `__iter__`, but verify
- Line 154: List comprehension over `dataT.shape` - should work due to `__iter__`, but verify
- Line 171: `list(iTList[0].shape)` - should work, but verify
- Line 181: Assignment `oTList[0].shape = [int(i) for i in newshape]` - should work (set_shape accepts list)
- Line 235: Assignment `op.attrs['out_shape'] = [int(i) for i in op.attrs['out_shape']]` - should work
- Line 236: Assignment `oTList[0].shape = op.attrs['out_shape']` - should work
- Line 266: List comprehension over `iTList[0].shape` - should work due to `__iter__`, but verify
- Line 293: `list(input_shape)` - should work, but verify
- Line 346: Comparison `x.shape[dim] != iTList[0].shape[dim]` - should work due to `__getitem__`
- Line 350: `list(iTList[0].shape)` - should work, but verify
- Line 351: `sum(x.shape[axis] for x in iTList)` - should work due to `__getitem__`
- Line 404: List comprehension over `input_shape` - should work due to `__iter__`, but verify
- Line 408: `input_shape[idx]` - should work due to `__getitem__`
- Line 446: `A.shape[axis]` - should work due to `__getitem__`
- Line 454: **CRITICAL**: `A.shape.copy()` - Shape doesn't have `.copy()` method! Need to use `Shape(A.shape)` or `list(A.shape)`
- Line 455: `tout_shape[axis] = split[tout_idx]` - Shape supports `__setitem__`, so this will work if `tout_shape` is a Shape object. However, since `.copy()` doesn't exist, we need to create a new Shape: `tout_shape = Shape(A.shape)` or use a list: `tout_shape = list(A.shape)`. If using a list, the assignment on line 462 will convert it to Shape via `set_shape()`.
- Line 492: **CRITICAL**: List concatenation `data_shape[:axis] + indicesT.shape + data_shape[axis + 1:]` - need to convert to lists first
- Line 512: `A.shape[start:end]` - should work due to `__getitem__` (slicing)
- Line 525: List comprehension `[iTList[0].shape[i] for i in perms]` - should work due to `__getitem__`

**Required Changes:**
```python
# Line 454: Change
tout_shape = A.shape.copy()
# To:
tout_shape = list(A.shape)  # or Shape(A.shape) if you need Shape object

# Line 492: Change
oTList[0].shape = data_shape[:axis] + indicesT.shape + data_shape[axis + 1:]
# To:
data_list = list(data_shape)
indices_list = list(indicesT.shape)
oTList[0].shape = data_list[:axis] + indices_list + data_list[axis + 1:]
```

### 4. `ttsim/front/functional/tensor_op.py`

**Issues:**
- Line 372: **CRITICAL**: List concatenation `shape[:start_dim] + [flat_size] + shape[end_dim+1:]` - need to convert shape to list first

**Required Changes:**
```python
# Line 372: Change
new_shape = shape[:start_dim] + [flat_size] + shape[end_dim+1:]
# To:
shape_list = list(shape)
new_shape = shape_list[:start_dim] + [flat_size] + shape_list[end_dim+1:]
```

### 5. `ttsim/graph/wl_graph.py`

**Issues:**
- Line 251: `tuple([int(d) for d in tval.shape])` - should work due to `__iter__`, but verify

**Required Changes:**
No changes needed if `__iter__` is properly implemented (which it is).

### 6. `ttsim/ops/tensor.py`

**Issues:**
- Line 18: `list(shape._shape)` - This is in the Shape constructor itself, so it's fine (internal implementation)
- Line 62: `[1] * (rank - current_rank) + self._shape` - This is internal, fine
- Line 66: `self._shape[-rank:]` - This is internal, fine

**Required Changes:**
No changes needed - these are internal to the Shape class.

## Testing Recommendations

After making changes, test the following scenarios:

1. **Shape creation and basic operations:**
   - Creating Shape from list/tuple
   - Creating Shape from another Shape
   - Indexing: `shape[0]`, `shape[-1]`
   - Slicing: `shape[:2]`, `shape[1:3]`
   - Length: `len(shape)`
   - Iteration: `for dim in shape:`
   - Comparison: `shape == [1, 2, 3]`

2. **Shape in tensor operations:**
   - Reshape operations
   - Concatenation operations
   - Transpose operations
   - Slice operations
   - Gather operations

3. **Shape conversions:**
   - `list(shape)` should work
   - `tuple(shape)` should work
   - `shape.view()` should return a list

## Priority

**High Priority (Will cause runtime errors):**
- `ttsim/ops/desc/tensor.py:454` - `.copy()` method doesn't exist
- `ttsim/ops/desc/tensor.py:492` - List concatenation with Shape
- `ttsim/front/functional/tensor_op.py:372` - List concatenation with Shape
- `ttsim/front/ttnn/tensor.py:266, 277` - List concatenation with Shape

**Medium Priority (Direct access to private member):**
- All instances of `._shape` access in `ttsim/front/ttnn/ttnn_shim.py`
- `ttsim/front/ttnn/tensor.py:237, 308, 326, 328`

**Low Priority (Should work but verify):**
- List comprehensions and iterations over shape (should work due to `__iter__`)
- Indexing operations (should work due to `__getitem__`)
- Comparisons (should work due to `__eq__`)

## Notes

- The Shape class implements `__iter__`, `__getitem__`, `__len__`, and `__eq__`, so many operations should work without changes
- The main issues are:
  1. Direct access to private `_shape` member
  2. List concatenation operations that assume Shape is a list
  3. Use of `.copy()` method which doesn't exist on Shape
