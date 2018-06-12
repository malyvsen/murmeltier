from ..unit import Unit
from murmeltier.utils import spaces


class ToSpace(Unit):
    '''
    Converts an np.array of np.float64 to a point in an OpenAI Gym space
    '''
    def __init__(self, in_specs = None, out_specs = None, **kwargs):
        if out_specs is None:
            raise ValueError('Provide out_specs')
        target_in_specs = spaces.size(out_specs)
        if in_specs is None:
            in_specs = target_in_specs
        if in_specs != target_in_specs:
            raise ValueError('in_specs must match size of out_specs')
        self.config(in_specs = in_specs, out_specs = out_specs)


    def get_output(self, input):
        return spaces.from_array(input, space = self.out_specs)
