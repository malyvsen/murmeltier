from ..unit import Unit


class Stack(Unit):
    '''
    Several units stacked on top of each other
    The output of each sub-unit is fed to the next one as input
    '''
    def __init__(self, in_specs = None, out_specs = None, units = None, unit_types = None, unit_type = None, specs = None, hidden_specs = None, **kwargs):
        '''
        units - an ordered list of sub-units, can be fed instead of params
        unit_types - provide this if you want the sub-units constructed for you
        unit_type - provide this instead of unit_types if you want all units to be identical
        specs - a full list of all the in/out specs in the stack
        hidden_specs - specs without the stack's own in_specs and out_specs
        '''
        if sum((units is not None, unit_types is not None, unit_type is not None)) != 1:
            raise ValueError('Provide exactly one of: units, unit_types, unit_type')

        if unit_types is not None or unit_type is not None:
            # the user wants the sub-units constructed for them
            if sum((specs is not None, hidden_specs is not None)) != 1:
                raise ValueError('Provide exactly one of: specs, hidden_specs')
            if hidden_specs is not None:
                if in_specs is None or out_specs is None:
                    raise ValueError('Provide in_specs and out_specs when providing hidden_specs')
                specs = [in_specs] + hidden_specs + [out_specs]

        if unit_type is not None:
            unit_types = [unit_type for i in range(len(specs) - 1)]
        if unit_types is not None:
            units = [unit_types[i](in_specs = specs[i], out_specs = specs[i + 1], **kwargs) for i in range(len(unit_types))]

        if in_specs is None:
            in_specs = units[0].in_specs
        if in_specs != units[0].in_specs:
            raise ValueError('in_specs must match the in_specs of the first layer')
        if out_specs is None:
            out_specs = units[-1].out_specs
        if out_specs != units[-1].out_specs:
            raise ValueError('out_specs must match the out_specs of the last layer')

        for i in range(len(units) - 1):
            if units[i].out_specs != units[i + 1].in_specs:
                raise ValueError('in and out specs of neighboring sub-units must match')

        self.config(in_specs = in_specs, out_specs = out_specs)
        self.params = {i: units[i] for i in range(len(units))}


    def get_output(self, input):
        value = input
        for i in range(len(self.params)):
            value = self.params[i].get_output(value)
        return value
