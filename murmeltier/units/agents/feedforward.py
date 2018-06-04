from ..unit import Unit
from ..structural import Stack
from ..spaces import FromSpace, ToSpace


class Feedfoward(Stack):
    '''
    A feedfoward agent for interacting with any OpenAI Gym environment
    '''
    def __init__(self, in_specs = None, out_specs = None, params = None, env = None, hidden_units = None, hidden_unit_types = None, hidden_unit_type = None, hidden_specs = None, stddevs = None, stddev = None):
        if in_specs is None:
            if env is None:
                raise ValueError('Provide env when not providing in_specs')
            in_specs = env.observation_space
        if in_specs != env.observation_space:
            raise ValueError('in_specs must match env.observation_space')

        if out_specs is None:
            if env is None:
                raise ValueError('Provide env when not providing out_specs')
            out_specs = env.action_space
        if out_specs != env.action_space:
            raise ValueError('out_specs must match env.action_space')

        if params is not None:
            if len(params) != 3:
                raise ValueError('params should have exactly 3 keys: one for from_space, one for hidden stack and one for to_space')
            Stack.__init__(self, in_specs = in_specs, out_specs = out_specs, params = params)
            return

        from_space = FromSpace(in_specs = in_specs)
        hidden = Stack(in_specs = in_specs, out_specs = out_specs, units = hidden_units, unit_types = hidden_unit_types, unit_type = hidden_unit_type, hidden_specs = hidden_specs, stddevs = stddevs, stddev = stddev)
        to_space = ToSpace(out_specs = out_specs)
        Stack.__init__(self, in_specs = in_specs, out_specs = out_specs, units = [from_space, hidden, to_space])
