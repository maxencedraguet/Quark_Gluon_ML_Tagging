#############################################################################
#
# BasicRunner.py
#
# Mother class of runner. Aimed at interacting with the main and a ML network
#
# Author -- Maxence Draguet (19/05/2020)
#
#############################################################################

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict

class _BaseRunner(ABC):
    def __init__(self, config: Dict) -> None:
        pass
    @abstractmethod
    def _extract_parameters(self, config: Dict) -> None:
        raise NotImplementedError("Base class method")

    def checkpoint_df(self, step: int) -> None:
        raise NotImplementedError("Base class method")

    def _setup_dataset(self, config: Dict):
        raise NotImplementedError("Base class method")

    def train(self):
        raise NotImplementedError("Base class method")

