'''
Units are the basic building block of any murmeltier AI
Each unit takes an input and produces an output with given specifications (usually just the dimensions)
In fact, any murmeltier AI is a unit in itself, because many units can be packed into one

Units keep their learned parameters inside, so sharing them is as simple as using the same unit instance many times
Units are vectorized, ie they can be added, subtracted, multiplied etc like vectors whose coordinates are their parameters
'''


__all__ = ['activations', 'agents', 'lin', 'spaces', 'structural']
