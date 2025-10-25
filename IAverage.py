from abc import ABC, abstractmethod

class IAverage(ABC):
    @abstractmethod   
    def add_value(self, value):
        pass
    @abstractmethod   
    def get_average(self):
        pass
    @abstractmethod
    def get_variance(self):
        pass
    @abstractmethod
    def reset(self):
        pass
