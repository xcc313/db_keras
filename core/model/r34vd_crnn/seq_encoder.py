
class EncoderWithReshape(object):
    def __init__(self):
        super(EncoderWithReshape, self).__init__()
    def __call__(self, inputs):
        None

class EncoderWithRNN(object):
    def __init__(self):
        super(EncoderWithRNN, self).__init__()
    def __call__(self, inputs):
        None


class SequenceEncoder(object):
    def __init__(self, params):
        super(SequenceEncoder, self).__init__()
        self.encoder_reshape = EncoderWithReshape(params)
        self.encoder = EncoderWithRNN(params)
    def __call__(self, inputs):
        encoder_features = self.encoder(inputs)