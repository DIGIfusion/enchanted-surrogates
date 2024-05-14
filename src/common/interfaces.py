from enum import Enum

class S(Enum):
    # Samplers ENUM
    SEQUENTIAL=0
    BATCH=1 

"""

@property
@abstractmethod
def model_interface(self) -> List[M]:
    # A list for storing what model interfaces the training method is compatible with
    raise NotImplementedError("model_interface not set!")

@property 
@abstractmethod
def sampler_interface(self) -> S: 
    raise NotImplementedError('sampler_interface is not set')
"""