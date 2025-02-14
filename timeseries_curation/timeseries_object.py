import numpy as np 

class TS_Obj():
    def __init__(self, **kwargs):
        pass

    def generate(self, length) -> np.array:
        pass

class LinearTrend(TS_Obj):
    def __init__(self, **kwargs):
        self.trend_level = kwargs.get('trend_level', None)
        if self.trend_level is None:
            raise ValueError('Trend level must be provided')

    def generate(self, length: int) -> np.array:
        return np.linspace(0, self.trend_level, length)
    
class ExponentialTrend(TS_Obj):
    def __init__(self, **kwargs):
        self.trend_level = kwargs.get('trend_level', None)
        if self.trend_level is None:
            raise ValueError('Trend level must be provided')

    def generate(self, length: int) -> np.array:
        start_value = 1  # Start of the exponential trend
        coefficient = np.log(self.trend_level / start_value)
        return np.exp(np.linspace(0, coefficient, length))
    
class LogTrend(TS_Obj):
    def __init__(self, **kwargs):
        """
        """
        self.trend_level = kwargs.get('trend_level', 0.1)
        
    def generate(self, length: int) -> np.array:
        linear_trend = np.linspace(0, abs(self.trend_level), length)
        logs = np.log(linear_trend + 1)
        if self.trend_level < 0:
            logs = -logs
        return logs
    
class Constant(TS_Obj):
    def __init__(self, **kwargs):
        self.value = kwargs.get('value', None)
        if self.value is None:
            raise ValueError('Constant mean value must be provided')

    def generate(self, length):
        return np.array([self.value] * length)

#########################Noise#########################
class GaussianWhiteNoise(TS_Obj):
    def __init__(self, **kwargs):
        self.noise_level = kwargs.get('noise_level', 1)

    def generate(self, length: int) -> np.array:
        return np.random.normal(0, self.noise_level, length)

class RedNoise(TS_Obj):
    def __init__(self, **kwargs):
        self.noise_level = kwargs.get('noise_level', 1)

    def generate(self, length: int) -> np.array:
        # # Generate white noise
        # white_noise = np.random.normal(0, self.noise_level, length)
        
        # # Integrate white noise to get Brownian motion
        # red_noise = np.cumsum(white_noise)
        
        # # Normalize to the desired noise level
        # red_noise -= np.mean(red_noise)
        # red_noise /= np.std(red_noise)
        # red_noise *= self.noise_level

        #Note: this method helps to generate a more realistic red noise; above might destroy increasing variance
        #in some cases
        white_noise = np.random.normal(0, self.noise_level / np.sqrt(length), length)
        red_noise = np.cumsum(white_noise)
        
        return red_noise
#########################Noise#########################
#########################Cycle#########################
class SineWave(TS_Obj):
    def __init__(self, **kwargs):
        self.amplitude = kwargs.get('amplitude', None)
        self.period = kwargs.get('period', None)
        
        if self.amplitude is None:
            raise ValueError('Amplitude must be provided')
        if self.period is None:
            raise ValueError('Period must be provided')

    def generate(self, length: int) -> np.array:
        #generate noise
        ts = self.amplitude * np.sin(2 * np.pi * np.arange(length) / self.period )
        return ts
    
class SawtoothWave(TS_Obj):
    def __init__(self, **kwargs):
        self.amplitude = kwargs.get('amplitude', None)
        self.period = kwargs.get('period', None)
        
        if self.amplitude is None:
            raise ValueError('Amplitude must be provided')
        if self.period is None:
            raise ValueError('Period must be provided')

    def generate(self, length: int) -> np.array:
        # Generate a sawtooth wave
        ts = 2 * self.amplitude * (np.arange(length) % self.period) / self.period - self.amplitude
        return ts
    
class SquareWave(TS_Obj):
    def __init__(self, **kwargs):
        self.amplitude = kwargs.get('amplitude', None)
        self.period = kwargs.get('period', None)
        
        if self.amplitude is None:
            raise ValueError('Amplitude must be provided')
        if self.period is None:
            raise ValueError('Period must be provided')

    def generate(self, length: int) -> np.array:
        # Generate a square wave
        ts = self.amplitude * np.sign(np.sin(2 * np.pi * np.arange(length) / self.period))
        return ts
    
class HarmonicOscillator(TS_Obj):
    def __init__(self, **kwargs):
        self.amplitude = kwargs.get('amplitude', None)
        self.frequency = kwargs.get('frequency', None)
        self.damping_factor = kwargs.get('damping_factor', None)

        if self.amplitude is None:
            raise ValueError('Amplitude must be provided')
        if self.frequency is None:
            raise ValueError('Frequency must be provided')
        if self.damping_factor is None:
            raise ValueError('Damping factor must be provided')

    def generate(self, length: int) -> np.array:
        t = np.arange(length)
        ts = self.amplitude * np.exp(-self.damping_factor * t) * np.cos(2 * np.pi * self.frequency * t)
        return ts
#########################Cycle#########################
class MovingAverage(TS_Obj):
    def __init__(self, **kwargs):
        self.num_past_values = kwargs.get('num_past_values', None)
        self.coef = kwargs.get('coef', None)
        self.scale = kwargs.get('scale', 1)
        self.noise = kwargs.get('noise', None)

        #make sure none of the required parameters are missing
        if self.num_past_values is None:
            raise ValueError('Number of past values must be provided')
        if self.coef is None:
            raise ValueError('Coefficient must be provided')
        if self.noise is None:
            raise ValueError('Noise generator must be provided')

    def generate(self, length):
        if length < self.num_past_values:
            raise ValueError('Length of time series must be greater than number of past values')
        #generate start values for the time series
        noise_process = self.noise.generate(length + self.num_past_values + 1)

        #generate using MA process
        ts = [] 
        #MA(q) = scale + noise + sum(coef[i] * noise[i]) for i in range(1, q+1)
        for i in range(self.num_past_values+1, len(noise_process)):
            ts.append(np.dot(noise_process[i-self.num_past_values:i], self.coef) + self.scale + noise_process[i])
        ts = np.array(ts)
        return ts
    
class AutoRegressive(TS_Obj):
    def __init__(self, **kwargs):
        self.num_past_values = kwargs.get('num_past_values', None)
        self.coef = kwargs.get('coef', None)
        self.scale = kwargs.get('scale', 1)
        self.noise = kwargs.get('noise', None)

        #make sure none of the required parameters are missing
        if self.num_past_values is None:
            raise ValueError('Number of past values must be provided')
        if self.coef is None:
            raise ValueError('Coefficient must be provided')
        if self.noise is None:
            raise ValueError('Noise generator must be provided')

    def generate(self, length):
        if length < self.num_past_values:
            raise ValueError('Length of time series must be greater than number of past values')
        #generate start values for the time series
        noise_process = self.noise.generate(length + self.num_past_values)
        start_values = self.scale + noise_process[:self.num_past_values]
        ts = start_values.tolist()

        #generate using AR process
        for i in range(length):
            ts.append(np.dot(ts[-self.num_past_values:], self.coef) + self.scale * noise_process[i+self.num_past_values])
        ts = np.array(ts[-length:])

        return ts 
    
    def check_stationary_ar(self):
        """
        Check if the AR(p) process is stationary. 
        A stationary AR(p) process requires that the roots of the characteristic polynomial lie outside the unit circle.
        """
        p = len(self.coef)
        if p == 0:
            return True
        ar_poly = np.r_[1, -np.array(self.coef)]
        
        return np.all(np.abs(np.roots(ar_poly)) > 1.0)
    
####################################pair ts objects####################################
class Pair_TS_Obj():
    def __init__(self, **kwargs):
        pass

    def generate(self, length) -> np.array:
        pass

class LaggedPair(Pair_TS_Obj):
    def __init__(self, **kwargs):
        self.base_ts = kwargs.get('base_ts', None)
        self.lagging_steps = kwargs.get('lagging_steps', None)
        self.base_ts_kwargs = kwargs.get('base_ts_kwargs', None)
        self.swap = kwargs.get('swap', False)
        
        if self.base_ts is None:
            raise ValueError('Base time series object must be provided')
        if self.lagging_steps is None:
            raise ValueError('Lagging steps must be provided')
        if self.base_ts_kwargs is None:
            raise ValueError('Base time series kwargs must be provided')

    def generate(self, length: int) -> np.array:
        actual_length = length + self.lagging_steps

        # Generate the base time series
        base_ts = self.base_ts(**self.base_ts_kwargs)
        ts = base_ts.generate(actual_length)

        # Generate the lagged time series
        lagged_ts = ts[self.lagging_steps:]
        assert len(lagged_ts) == length == len(ts[:length])

        if self.swap:
            return lagged_ts, ts[:length]
        
        return ts[:length], lagged_ts
    
class LaggedPairWithAmplitude(LaggedPair):
    def __init__(self, **kwargs):
        self.base_ts = kwargs.get('base_ts', None)
        self.lagging_steps = kwargs.get('lagging_steps', None)
        self.base_ts_kwargs = kwargs.get('base_ts_kwargs', None)
        self.swap = kwargs.get('swap', False)
        self.amplitude = kwargs.get('amplitude', 1)
        
        if self.base_ts is None:
            raise ValueError('Base time series object must be provided')
        if self.lagging_steps is None:
            raise ValueError('Lagging steps must be provided')
        if self.base_ts_kwargs is None:
            raise ValueError('Base time series kwargs must be provided')

    def generate(self, length: int) -> np.array:
        actual_length = length + self.lagging_steps

        # Generate the base time series
        base_ts = self.base_ts(**self.base_ts_kwargs)
        ts = base_ts.generate(actual_length)

        # Generate the lagged time series
        lagged_ts = ts[self.lagging_steps:]
        assert len(lagged_ts) == length == len(ts[:length])

        if self.swap:
            return lagged_ts, ts[:length]
        
        return ts[:length], lagged_ts * self.amplitude
    
class GrangerPair(Pair_TS_Obj):
    def __init__(self, **kwargs):
        self.base_ts = kwargs.get('base_ts', None)
        self.base_ts_kwargs = kwargs.get('base_ts_kwargs', None)
        self.p = kwargs.get('p', None)
        self.q = kwargs.get('q', None)
        self.granger_coef = kwargs.get('granger_coef', None)
        self.ar_coef = kwargs.get('ar_coef', None)
        self.ar_steps = kwargs.get('ar_steps', None)
        self.noise = kwargs.get('noise', None)
        self.swap = kwargs.get('swap', False)
        
        if self.base_ts is None:
            raise ValueError('Base time series object must be provided')
        if self.base_ts_kwargs is None:
            self.base_ts_kwargs = {}
        if self.p is None:
            raise ValueError('Granger lagging steps must be provided')
        if self.q is None:
            raise ValueError('Number of AR steps must be provided')
        if self.granger_coef is None:
            raise ValueError('Granger coefficient must be provided')
        if self.ar_coef is None:
            raise ValueError('AR coefficient must be provided')
        if self.ar_steps is None:
            raise ValueError('Number of AR steps must be provided')
        if self.noise is None:
            raise ValueError('Noise generator must be provided')
        
        assert len(self.granger_coef) == (self.q - self.p) 
        assert len(self.ar_coef) == self.ar_steps
        assert self.p > 1

    def generate(self, length: int) -> np.array:
        actual_length = length + self.q

        base_ts = self.base_ts(**self.base_ts_kwargs)
        ts = base_ts.generate(actual_length)
        noise = self.noise.generate(actual_length)

        # initialized the lagged time series
        lagged_ts = np.zeros(length)
        lagged_ts[:self.ar_steps] = ts[:self.ar_steps]

        # Generate the Granger time series
        for i in range(self.q+self.ar_steps, actual_length):
            try:
                lagged_ts[i - self.q] = np.dot(ts[i - self.q:i - self.p], self.granger_coef) + np.dot(lagged_ts[i - self.q - self.ar_steps:i - self.q], self.ar_coef) + noise[i]
            except Exception as e:
                raise ValueError(f'Error in Granger pair generation: {e}')

        if self.swap:
            return lagged_ts, ts[:length]
        
        return ts[:length], lagged_ts
