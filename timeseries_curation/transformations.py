from timeseries_curation.inject_anomalies import InjectAnomalies
from timeseries_curation.anomaly_parameters import ANOMALY_PARAM_GRID

import numpy as np 
from sklearn.model_selection import ParameterGrid

class Transformation():
    def __init__(self):
        pass

    def transform(self, ts):
        pass 

#anomaly injections 
class Anomaly(Transformation):
    def __init__(self, **kwarg):
        self.max_window_size = kwarg.get('max_window_size', 20)
        self.min_window_size = kwarg.get('min_window_size', 8)
        self.anomaly_name = kwarg.get('anomaly_name', None)
        if self.anomaly_name is None:
            raise ValueError('Anomaly name must be provided')
        self.anomaly_params = ANOMALY_PARAM_GRID[self.anomaly_name]
        self.injector = InjectAnomalies(random_state=42, 
                                        verbose=False,
                                        max_window_size=self.max_window_size,
                                        min_window_size=self.min_window_size,)
    def transform(self, ts):
        for ad_param in list(ParameterGrid(self.anomaly_params)):
            ad_param['scale'] = ad_param['scale'] * np.std(ts)
            ts = ts.reshape(1, -1)
            ad_param['T'] = ts
            break

        ts_a, _, _ = self.injector.inject_anomalies(**ad_param)
        return ts_a[0]

class SignFlip(Transformation):
    def __init__(self, **kwarg):
        pass

    def transform(self, ts):
        return -ts