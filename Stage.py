from Map import *
import enum
import copy

class status(enum.Enum):
    SUCCESS = 0
    FAILURE = 1

class Stage(object):
    def __init__(self, input_keys=[], output_keys=[]):
        self._input = set(input_keys)
        self._output = set(output_keys)

    def get_registered_input_keys(self):
        return copy.deepcopy(self._input)

    def get_registered_output_keys(self):
        return copy.deepcopy(self._output)

    def execute(self, io, static_io):
        raise NotImplementedError("[Stage] Needs to be implemented by user")