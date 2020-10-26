from keras import layers

class EncoderWithRNN(object):
    def __init__(self, params):
        super(EncoderWithRNN, self).__init__()
        self.rnn_hidden_size = params['hidden_size']
    def __call__(self, inputs):
        lstm_list = []
        name_prefix = "lstm"
        rnn_hidden_size = self.rnn_hidden_size
        for no in range(1, 3):
            if no == 1:
                is_reverse = False
            else:
                is_reverse = True
            name = "%s_st1_fc%d" % (name_prefix, no)
            fc = layers.Dense(rnn_hidden_size * 4)(inputs)
            name = "%s_st1_out%d" % (name_prefix, no)



class SequenceEncoder(object):
    def __init__(self, params):
        super(SequenceEncoder, self).__init__()
        self.encoder = EncoderWithRNN(params)
    def __call__(self, inputs):
        encoder_features = layers.Flatten()(inputs)
        encoder_features = self.encoder(inputs)
        return encoder_features