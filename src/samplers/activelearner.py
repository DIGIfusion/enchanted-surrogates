from common import S
from .base import Sampler 
from typing import Dict


class ActiveLearner(Sampler): 
    sampler_interface = S.ACTIVE # NOTE: This could likely also be batch...
    def __init__(self, bounds, num_samples, parameters, **AL_kwargs): 
        self.total_budget = num_samples 
        self.AL_kwargs = AL_kwargs
        pass 
    
    def get_initial_parameters(self):
        # random samples 
        pass 

    def get_next_parameter(self, model, train, pool) -> Dict[str, float]:
        # pseudocode, fix later
        train = TensorFeatureData(train.x)
        pool = TensorFeatureData(pool.x)
        y_train = train.y
        feature_data =  {'train': train, 'pool': pool}
        new_idxs, al_stats = select_batch(models=[model], data=feature_data, y_train=y_train,
                                        selection_method='lcmd', sel_with_train=True, 
                                        base_kernel='grad', kernel_transforms=[('rp', [512])])
        return new_idxs
    
    def update_pool_and_train(self):
        pass
