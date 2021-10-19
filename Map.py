import threading
import sys

sys.tracebacklimit = 0

__all__ = ['Map', 'MapAccess']

def get_const(obj):
    if hasattr(obj, '__dict__'):
        return Const(obj)
    else:
        return obj

class Const(object):
    def __init__(self, obj):
        self._obj = obj
        self.__initialized = True

    def __getattr__(self, name):
        attr = getattr(self._obj,name)
        return get_const(attr)

    def __getitem__(self, name):
        attr = self._obj[name]
        return get_const(attr)

    def __setattr__(self, name, value):
        if '_const__initialized' not in self.__dict__: 
            return object.__setattr__(self, name, value)
        raise TypeError()

    def __delattr__(self, name):
        raise TypeError()

class Map(object):
	def __init__(self):
		super(Map, self).__setattr__('_map', dict())
		super(Map, self).__setattr__('_locks', dict())

	def __getattr__(self, key):
		if key[0] == '_':
			return object.__getattr__(self, key)

		if not key in self._locks.keys():
			self._locks[key] = threading.Lock()

		try:
			with self._locks[key]:
				return self._map[key]
		except:
			print("[Map] Key '{}' is not available. Available keys are : {}".format(key, str(list(self._map.keys()))))

	def __setattr__(self, key, value):
		if key[0] == '_':
			return object.__setattr__(self, key, value)

		if not key in self._locks.keys():
			self._locks[key] = threading.Lock()

		self._locks[key].acquire()
		self._map[key] = value
		self._locks[key].release()

class MapAccess(object):
	def __init__(self, mapObj, input_keys=[], output_keys=[]):
		super(MapAccess, self).__setattr__('_mapObj', mapObj)
		super(MapAccess, self).__setattr__('_input', set(input_keys))
		super(MapAccess, self).__setattr__('_output', set(output_keys))
		
		self.__initialized = False

	def __getattr__(self, key):
		if key[0] == '_':
			return object.__getattr__(self, key)
			
		if key not in self._input:
			raise AttributeError("[Map Access] '{}' is not declared in inputs, so read permission denied. [inputs: {}]".format(key, self._input))
		
		if key not in self._output:
			return get_const(getattr(self._mapObj, key))

		return getattr(self._mapObj, key)

	def __setattr__(self, key, value):
		if '_MapAccess__initialized' not in self.__dict__:
			return object.__setattr__(self, key, value)
		elif key in ['_mapObj', '_input', '_output'] and ('_MapAccess__initialized' in self.__dict__):
			raise AttributeError("[MapAcess] '{}' is read-only attribute".format(key))

		if key not in self._output:
			raise AttributeError("[Map Access] '{}' is not declared in outputs, so write permission denied. [outputs: {}]".format(key, self._output))

		setattr(self._mapObj, key, value)
