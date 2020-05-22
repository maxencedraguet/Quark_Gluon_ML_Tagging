#############################################################################
#
# BaseDataLoader.py
#
# A base for data loader
#
# Author -- Maxence Draguet (19/05/2020)
#
#############################################################################
from abc import ABC, abstractmethod
from typing import Dict

class _BaseDataLoader(ABC):

    def __init__(self, config: Dict) -> None:
        pass
    
    @abstractmethod
    def load_separate_data(self):
        raise NotImplementedError("Base class method")
