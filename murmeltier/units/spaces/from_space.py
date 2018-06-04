from ..unit import Unit
from murmeltier.utils import spaces


class FromSpace(Unit):
    '''
    Converts a point in an OpenAI Gym space to an np.array of np.float64
    '''
    def __init__(self, in_specs, out_specs = None, params = None):
        if out_specs is None:
            out_specs = spaces.size(in_specs)
        Unit.construct(self, in_specs = in_specs, out_specs = out_specs)


    def get_output(self, input):
        return space.to_array(input, space = self.space)
