# Murmeltier
German for marmot. Also, a Python framework for AI experiments. Flexible, not necessarily fast - researching differences of kind rather than differences of scale.

**murmeltier is no longer in development, as I have discovered the existence of PyTorch which does exactly the same, but better :)**

## Installation
`pip install murmeltier`  
*...soon enough.*

## Dependencies
Python 3.6.5  
NumPy 1.14.3  
gym 0.10.5  

## Quick guide
### Units
Units are the basic building block of any murmeltier AI.  
Each unit takes an input and produces an output with given specifications (usually just the dimensions).  
```python
>>> from murmeltier.units.lin import Weights
>>> weights = Weights(in_specs = 3, out_specs = 5, stddev = 1)
>>> weights.get_output([1, 2, 3])
array([ 1.58316566, -3.37014459,  0.74510459,  0.90593016,  0.46192767])
```

In fact, any murmeltier AI is a unit in itself, because many units can be packed into one.  
```python
>>> from murmeltier.units.lin import Biases
>>> from murmeltier.units.structural import Stack
>>> biases = Biases(in_specs = 5, stddev = 1)
>>> stack = Stack(units = [weights, biases])
>>> stack.get_output([1, 2, 3])
array([ 0.56071995,  6.83696213,  4.36535554, -6.04676206, -1.08439716])
```

Units keep their learned parameters inside, so sharing them within the architecture is as simple as using the same unit instance many times.  
Units are vectorized - they can be added, subtracted, multiplied etc like vectors whose coordinates are their parameters.  
```python
>>> other_stack = stack * 2
>>> other_stack.get_output([1, 2, 3])
array([  1.12143989,  13.67392426,   8.73071109, -12.09352411,  -2.16879431])
```

## Author
Nicholas Bochenski @malyvsen
