# Functional API Documentation

The `ttsim.front.functional` module provides a PyTorch-inspired functional interface for building neural networks in Polaris. It consists of two main components:

## 1. Operator Interface (`op.py`)

The operator interface provides a high-level API for creating and manipulating operators in the Polaris simulation framework.

### Core Classes

#### `SimOpHandle`
A wrapper class that manages operator information and execution:
- Handles input/output tensor management
- Provides parameter management
- Supports operator attribute validation
- Links operators to modules for hierarchical organization

#### `SplitOpHandle`
A specialized handle for the Split operator:
- Supports variable number of outputs
- Manages output tensor naming and linking

#### `VariadicInputOpHandle`
A handle for operators with variable number of inputs:
- Supports input range validation
- Useful for operators like Concat, Trilu, and Slice

### Available Operators

1. **Unary Operators**
   - `Identity`: Pass-through operator
   - `Tanh`: Hyperbolic tangent activation
   - `Softmax`: Softmax normalization
   - `Cast`: Data type conversion
   - `Shape`: Get tensor shape
   - `Transpose`: Matrix transposition
   - `Gelu`: Gaussian Error Linear Unit activation
   - `Relu`: Rectified Linear Unit activation
   - `LeakyReLU`: Leaky ReLU activation
   - `Sigmoid`: Sigmoid activation

2. **Binary Operators**
   - `Add`: Element-wise addition
   - `Mul`: Element-wise multiplication
   - `Gather`: Indexed tensor gathering
   - `MatMul`: Matrix multiplication
   - `Reshape`: Tensor reshaping
   - `Pow`: Element-wise power
   - `Unsqueeze`: Dimension addition
   - `Equal`: Element-wise equality comparison

3. **Ternary Operators**
   - `Where`: Conditional selection
   - `Range`: Sequence generation

4. **Neural Network Operators**
   - `Conv2d`: 2D convolution
   - `Linear`: Fully connected layer
   - `LayerNorm`: Layer normalization
   - `BatchNorm2d`: 2D batch normalization
   - `AveragePool2d`: 2D average pooling
   - `Resize`: Tensor resizing

5. **Variadic Input Operators**
   - `ConcatX`: Tensor concatenation (2 or more inputs)
   - `TriluX`: Triangle mask (1-2 inputs)
   - `SliceF`: Tensor slicing (3-6 inputs)

## 2. Module System (`sim_nn.py`)

The module system provides a PyTorch-like organization for building neural networks.

### Core Classes

#### `Module`
Base class for all neural network modules:
- Manages tensors, operators, and submodules
- Supports hierarchical organization
- Provides graph construction utilities
- Handles tensor and operator naming

Features:
- Automatic tensor registration
- Operator handle management
- Submodule organization
- Graph construction for forward passes
- String representation for debugging

#### `ModuleList`
Container for sequential organization of modules:
- Immutable after construction
- Supports indexing and iteration
- Ensures unique module names
- Maintains module ordering

### Usage Example

```python
class MyNetwork(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d("conv1", in_channels=3, out_channels=64, kernel_shape=[3,3])
        self.bn1 = BatchNorm2d("bn1", channels=64)
        self.relu = Relu("relu1")
        self.pool = AveragePool2d("pool1", kernel_shape=(2,2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# Create network
net = MyNetwork()

# Create input tensor
input_tensor = create_shape_tensor("input", [1, 3, 32, 32])

# Run forward pass
output = net(input_tensor)
```

## Integration with Shape Inference

The functional API integrates seamlessly with Polaris's shape inference system:

1. **Automatic Shape Inference**
   - Each operator handle manages its shape inference
   - Shape validation during operator construction
   - Efficient memory usage by avoiding tensor allocation

2. **Performance Statistics**
   - Operators track performance metrics
   - Statistics available through operator handles
   - Useful for optimization and analysis

3. **Graph Construction**
   - Automatic graph building from module hierarchy
   - Proper tensor and operator linking
   - Support for complex network architectures

## Best Practices

1. **Memory Efficiency**
   - Use shape inference when possible
   - Avoid unnecessary tensor allocations
   - Leverage operator handles for tensor management

2. **Module Organization**
   - Group related operations into modules
   - Use meaningful names for operators and tensors
   - Maintain clean module hierarchies

3. **Error Handling**
   - Check required attributes for operators
   - Validate input shapes and types
   - Use appropriate operator handles for different cases

4. **Performance Optimization**
   - Monitor operator statistics
   - Use appropriate operator variants
   - Consider tensor layout and memory access patterns
