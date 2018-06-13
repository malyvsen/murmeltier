'''
Initializers are used to provide initial values for internal unit parameters
Each of the functions in this submodule returns another function, which in turn returns the value for some parameter of a unit
The first function should always take the keyword argument:
* shape - shape of the np.array that the second function returns
The second function should take any keword arguments needed for initialization, eg. the standard deviation
Currying is optional, but recommended
'''


__all__ = ['constant', 'normal', 'uniform']


from .constant import constant
from .normal import normal
from .uniform import uniform
