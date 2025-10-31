from enchanted_surrogates.samplers.base_sampler import Sampler
import warnings
import pandas as pd

class CsvSampler(Sampler):
    def __init__(self, csv_path, budget=None, batch_size=None, parameters=None, **kwargs):
        self.df = pd.read_csv(csv_path)
        
        if budget is not None:
            self.budget = budget
        else:
            self.budget = len(self.df)
        
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            self.batch_size = self.budget
            
        if parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = self.df.columns
        
        self.batch_number = 0
        self.samples = self.df[self.parameters].to_dict(orient='records')
                
    def get_next_samples(self) -> list[dict]:
        samples = self.samples[self.batch_number*self.batch_size  :  min((self.batch_number+1)*self.batch_size, self.budget)]
        self.batch_number += 1
        self.submitted += len(samples)
        return samples
        
    def register_future(self, future):
        """ Doesn't matter for random sampler TODO: Probably? """
        return None

    def register_futures(self, futures):
        return None