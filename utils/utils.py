import json 
import numpy as np 
import random 
from dataclasses import dataclass 
from typing import Union, List, Optional

from timeseries_curation.timeseries_object import (
    TS_Obj,
    LinearTrend,
    ExponentialTrend,
    Constant,
    Pair_TS_Obj
)

from timeseries_curation.composer import (
    AdditiveComposer,
    MultiplicativeComposer, 
    ConcatenateComposer
)

from timeseries_curation.transformations import (
    Transformation
)

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

def jdump(obj, path):
    """
    Dump a JSON object to a file.
    """
    with open(path, 'wb') as f:
        json.dump(obj, f, indent=4)

######################################sampling methods for each option######################################
class Sampler:
    def sample(self):
        pass

class NumericalParameterSampler(Sampler):
    '''
    a sampler object that samples a numerical value from a given range
    '''
    def __init__(self, min, max, sampling_method='uniform'):
        self.min = min
        self.max = max
        self.sampling_method = sampling_method
    def sample(self) -> float:
        if self.sampling_method == 'uniform':
            return np.random.uniform(self.min, self.max)
        elif self.sampling_method == 'uniform_int':
            return np.random.randint(self.min, self.max)
        else:
            raise ValueError('Sampling method not supported')

class NumericalListParameterSampler(Sampler):
    '''
    a sampler object that samples a numerical value from a given list of length n (fixed value)
    '''
    def __init__(self, min, max, size, sampling_method='uniform'):
        self.min = min
        self.max = max
        self.sampling_method = sampling_method
        self.size = size
    def sample(self) -> np.array:
        if self.sampling_method == 'uniform':
            return np.random.uniform(self.min, self.max, self.size)
        elif self.sampling_method == 'uniform_int':
            return np.random.randint(self.min, self.max)
        else:
            raise ValueError('Sampling method not supported')
        
######################################data classes used by the templates######################################
@dataclass
class Option:
    option_name:str 
    timeseries_obj:Union[TS_Obj, List[Union[TS_Obj, Transformation]]]
    timeseries_obj_kwargs:Union[dict, List[dict]]
    num_draws: Optional[Union[int, None]] = None #if None, num of draws will be using the default value given by the user
    combine_method: Optional[str] = '' #combine method for multiple timeseries object, such as additive, multiplicative, concatenate
    noise_snr: Optional[float] = 0.1 #signal to noise ratio if noise is applied 

#generate 2 independent ts given two configs
@dataclass
class TwoTSOption:
    option_name:str 
    timeseries_obj1:Union[TS_Obj, List[Union[TS_Obj, Transformation]]]
    timeseries_obj_kwargs1:Union[dict, List[dict]]
    timeseries_obj2:Union[TS_Obj, List[Union[TS_Obj, Transformation]]]
    timeseries_obj_kwargs2:Union[dict, List[dict]]
    num_draws: Optional[Union[int, None]] = None
    combine_method: Optional[str] = ''
    noise_snr1: Optional[float] = 0.0
    noise_snr2: Optional[float] = 0.0

#generate 2 ts given 1 config (generation of two ts are not independent such as lagged pair and granger pair)
@dataclass
class PairTSOption:
    option_name:str 
    timeseries_obj:Pair_TS_Obj
    timeseries_obj_kwargs: dict
    num_draws: Optional[Union[int, None]] = None
    combine_method: Optional[str] = ''
    noise_snr: Optional[float] = 0.1

#provide so that the evaluated model can be informed with necessary context
@dataclass 
class Concept:
    concept_name:str 
    concept_description:str
    concept_example:Union[list, np.array]
    concept_example_string:str 
######################################data classes used by the templates######################################

def process_kwargs(kwargs: dict) -> dict:
    '''
    process the kwargs dictionary, if the value is a Sampler object, sample the value
    the resulting dictionary will have the same keys but the values will be the sampled values
    '''
    new_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, Sampler):
            new_kwargs[k] = v.sample()
        else:
            new_kwargs[k] = v
    return new_kwargs

def execute_option(option_str: str):
    if option_str.startswith('[') and option_str.endswith(']'):
        param_name, param_sampler = option_str[1:-1].split(':')
        param_sampler = eval(param_sampler)
        return param_name, round(param_sampler.sample(), 2)
    else:
        return option_str, option_str

def calculate_noise_level(snr: float, ts_obj: TS_Obj, length) -> float:
    '''
    compute the noise leval base on given signal to noise ratio and the timeseries object,
    we do this because the noise level should behave differently for different timeseries object
    '''
    if type(ts_obj) == LinearTrend:
        step_size = abs(ts_obj.trend_level) * length / length
        noise_level = step_size * snr
    elif type(ts_obj) == ExponentialTrend:
        step_size = ts_obj.trend_level / length 
        noise_level = max(step_size * snr, 0.5)
    elif type(ts_obj) == Constant:
        noise_level = abs(ts_obj.value) * snr / 10
    else:
        noise_level = snr
    return noise_level

def get_ts_obj_from_option(option: Option) -> TS_Obj:
    '''
    Given an option object, return the timeseries object
    '''
    #some sanity checks
    assert isinstance(option.timeseries_obj, list) == isinstance(option.timeseries_obj_kwargs, list), f'Type of timeseries_obj and timeseries_obj_kwargs must be the same'
    if isinstance(option.timeseries_obj, list):
        assert len(option.timeseries_obj) == len(option.timeseries_obj_kwargs), f'Length of timeseries_obj and timeseries_obj_kwargs must be the same'
        assert len(option.timeseries_obj) > 0, f'Length of timeseries_obj and kwargs must be greater than 0'
    
    #TODO: currently we only support one global transformation for all timeseries object, not each object has its own transformation
    if isinstance(option.timeseries_obj, list):
        timeseries_obj, timeseries_obj_kwargs = [], [] 
        transformation_obj, transformation_obj_kwargs = [], []

        for obj, kwarg in zip(option.timeseries_obj, option.timeseries_obj_kwargs):
            if issubclass(obj, TS_Obj):
                timeseries_obj.append(obj)
                timeseries_obj_kwargs.append(kwarg)
            elif issubclass(obj, Transformation):
                transformation_obj.append(obj)
                transformation_obj_kwargs.append(kwarg)
            else:
                raise ValueError(f'Invalid object type: {type(obj)}, must be either be a subclass of TS_Obj or Transformation')

        processed_ts_kwargs = [process_kwargs(kwargs) for kwargs in timeseries_obj_kwargs]
        processed_transformation_kwargs = [process_kwargs(kwargs) for kwargs in transformation_obj_kwargs]
        ts_objs = [ts_obj(**kwargs) for ts_obj, kwargs in zip(timeseries_obj, processed_ts_kwargs)]
        transformation_objs = [transformation_obj(**kwargs) for transformation_obj, kwargs in zip(transformation_obj, processed_transformation_kwargs)]

        if option.combine_method == 'choose_from':
            combined_object = random.choice(ts_objs)
        elif option.combine_method == 'additive':
            combined_object = AdditiveComposer(ts_objs)
        elif option.combine_method == 'multiplicative':
            combined_object = MultiplicativeComposer(ts_objs)
        elif option.combine_method == 'concatenate':
            combined_object = ConcatenateComposer(ts_objs)
        else:
            raise ValueError(f'Invalid combine method: {option.combine_method}, choose from [choose_from, additive, multiplicative, concatenate]')
        if len(transformation_objs) > 0:
            combined_object.transformations = transformation_objs
        return combined_object
    
    else:
        #if the timeseries_obj is a single object, just process the kwargs and return the object
        processed_kwargs = process_kwargs(option.timeseries_obj_kwargs)
        return option.timeseries_obj(**processed_kwargs)

def get_ts_obj_from_two_options(option: TwoTSOption) -> Union[TS_Obj, TS_Obj]:
    option1 = Option(option.option_name, option.timeseries_obj1, option.timeseries_obj_kwargs1, option.num_draws, option.combine_method, option.noise_snr1)
    option2 = Option(option.option_name, option.timeseries_obj2, option.timeseries_obj_kwargs2, option.num_draws, option.combine_method, option.noise_snr2)

    ts_obj1 = get_ts_obj_from_option(option1)
    ts_obj2 = get_ts_obj_from_option(option2)
    return ts_obj1, ts_obj2

def get_pair_ts_obj_from_option(option: PairTSOption) -> Pair_TS_Obj:
    '''
    for pair ts object, one timeseries object is used to generate two timeseries object
    the base timeseries object is created at runtime
    '''
    option.timeseries_obj_kwargs['base_ts_kwargs'] = process_kwargs(option.timeseries_obj_kwargs['base_ts_kwargs'])
    processed_kwargs = process_kwargs(option.timeseries_obj_kwargs)
    return option.timeseries_obj(**processed_kwargs)
