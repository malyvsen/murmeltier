from ..unit import Unit
from ..structural import Stack
from ..spaces import FromSpace, ToSpace


class Feedfoward(Unit):
    '''
    A feedfoward agent for interacting with any OpenAI Gym environment
    '''
    def __init__(self, in_specs = None, out_specs = None, env = None, hidden_units = None, hidden_unit_types = None, hidden_unit_type = None, hidden_specs = None, **kwargs):
        if env is None:
            if in_specs is None:
                raise ValueError('Provide in_specs when not providing env')
            if out_specs is None:
                raise ValueError('Provide out_specs when not providing env')
        else:
            if in_specs is None:
                in_specs = env.observation_space
            if in_specs != env.observation_space:
                raise ValueError('in_specs must match env.observation_space')
            if out_specs is None:
                out_specs = env.action_space
            if out_specs != env.action_space:
                raise ValueError('out_specs must match env.action_space')

        self.config(in_specs = in_specs, out_specs = out_specs)
        self.params['from_space'] = FromSpace(in_specs = in_specs, **kwargs)
        self.params['to_space'] = ToSpace(out_specs = out_specs, **kwargs)
        self.params['hidden'] = Stack(in_specs = self.params['from_space'].out_specs, out_specs = self.params['to_space'].in_specs, units = hidden_units, unit_types = hidden_unit_types, unit_type = hidden_unit_type, hidden_specs = hidden_specs, **kwargs)


    def get_output(self, input):
        pre_hidden = self.params['from_space'].get_output(input)
        post_hidden = self.params['hidden'].get_output(pre_hidden)
        return self.params['to_space'].get_output(post_hidden)
