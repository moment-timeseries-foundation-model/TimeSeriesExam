import numpy as np 

class Composer:
    def __init__(self, ts_objects):
        self.ts_objects = ts_objects
    
    def generate(self, length):
        pass

class AdditiveComposer(Composer):
    def __init__(self, ts_objects):
        self.ts_objects = ts_objects
    
    def generate(self, length):
        ts = 0 
        for ts_obj in self.ts_objects:
            ts_obj_generation = ts_obj.generate(length)
            assert len(ts_obj_generation) == length, f'Length of generated time series is not equal to the desired length {type(ts_obj)}'
            if type(ts_obj_generation) == list:
                ts_obj_generation = np.array(ts_obj_generation)
            ts += ts_obj_generation

        return ts
    
class MultiplicativeComposer(Composer):
    def __init__(self, ts_objects):
        self.ts_objects = ts_objects
    
    def generate(self, length):
        ts = 1 
        for ts_obj in self.ts_objects:
            ts_obj_generation = ts_obj.generate(length)
            assert len(ts_obj_generation) == length, f'Length of generated time series is not equal to the desired length {type(ts_obj)}'
            if type(ts_obj_generation) == list:
                ts_obj_generation = np.array(ts_obj_generation)
            ts *= ts_obj_generation

        return ts

class ConcatenateComposer(Composer):
    def __init__(self, ts_objects):
        self.ts_objects = ts_objects
    
    def generate(self, length):
        ts = [] 
        length_per_ts = length // len(self.ts_objects)
        for i, ts_obj in enumerate(self.ts_objects):
            if i == len(self.ts_objects) - 1:
                length_per_ts += length % len(self.ts_objects)
            ts_obj_generation = ts_obj.generate(length_per_ts) 
            if len(ts) != 0:
                ts_obj_generation += ts[-1][-1]
            ts.append(ts_obj_generation)
        ts = np.concatenate(ts)
        return ts