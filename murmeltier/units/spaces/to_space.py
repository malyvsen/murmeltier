from ..unit import Unit
from murmeltier.utils import spaces


class ToSpace(Unit):
    '''
    Converts an np.array of np.float64 to a point in an OpenAI Gym space
    '''
    def __init__(self, in_specs = None, out_specs = None, params = None):
        if out_specs is None:
            raise ValueError('Provide out_specs')
        if in_specs is None:
            in_specs = spaces.size(out_specs)
        Unit.construct(self, in_specs = in_specs, out_specs = out_specs)


    def get_output(self, input):
        return space.from_array(input, space = self.space)
