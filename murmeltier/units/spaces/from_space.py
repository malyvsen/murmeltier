from ..unit import Unit
from murmeltier.utils import spaces


class FromSpace(Unit):
    '''
    Converts a point in an OpenAI Gym space to an np.array of np.float64
    '''
    def __init__(self, in_specs, out_specs = None, **kwargs):
        target_out_specs = spaces.size(in_specs)
        if out_specs is None:
            out_specs = target_out_specs
        if out_specs != target_out_specs:
            raise ValueError('out_specs must match size of in_specs')
        self.config(in_specs = in_specs, out_specs = out_specs)


    def get_output(self, input):
        return spaces.to_array(input, space = self.in_specs)
