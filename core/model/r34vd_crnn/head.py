import tensorflow as tf
from keras import layers, models

class CTCHead(object):
    def __init__(self, params):
        self.__name__ = 'CTCHead'
        self.k = params['det']['k']



    def __call__(self, input):
        None
