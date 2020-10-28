from keras import layers


class SequenceEncoder(object):
    def __init__(self, params):
        super(SequenceEncoder, self).__init__()
        self.char_num = params.char_ops.get_char_num()
        self.rnn_hidden_size = params['hidden_size']
    def __call__(self, inputs):
        encoder_features = layers.TimeDistributed(layers.Flatten())(inputs)

        x = layers.Bidirectional(layers.LSTM(self.rnn_hidden_size, return_sequences=True, use_bias=True,
                                             recurrent_activation='sigmoid'))(encoder_features)
        x = layers.TimeDistributed(layers.Dense(self.rnn_hidden_size))(x)
        x = layers.Bidirectional(layers.LSTM(self.rnn_hidden_size, return_sequences=True, use_bias=True,
                                             recurrent_activation='sigmoid'))(x)
        fc= layers.TimeDistributed(layers.Dense(self.char_num + 1), name="DenseSoftmax")(x)
        return fc
