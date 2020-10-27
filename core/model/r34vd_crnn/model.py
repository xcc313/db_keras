from .backbone import ResNet
from .seq_encoder import SequenceEncoder

from keras import layers, models
import keras.backend as K
import tensorflow as tf


# Define CTC loss
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    # hack for load_model
    import tensorflow as tf

    ''' from TF: Input requirements
    1. sequence_length(b) <= time for all b
    2. max(labels.indices(labels.indices[:, 1] == b, 2)) <= sequence_length(b) for all b.
    '''

    # print("CTC lambda inputs / shape")
    # print("y_pred:",y_pred.shape)  # (?, 778, 30)
    # print("labels:",labels.shape)  # (?, 80)
    # print("input_length:",input_length.shape)  # (?, 1)
    # print("label_length:",label_length.shape)  # (?, 1)


    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def ctc_complete_decoding_lambda_func(args, **arguments):
    """
    Complete CTC decoding using Keras (function K.ctc_decode)
    :param args:
        y_pred, input_length
    :param arguments:
        greedy, beam_width, top_paths
    :return:
        K.ctc_decode with dtype='float32'
    """

    # import tensorflow as tf # Require for loading a model saved

    y_pred, input_length = args
    my_params = arguments

    assert (K.backend() == 'tensorflow')

    return K.cast(K.ctc_decode(y_pred, tf.squeeze(input_length), greedy=my_params['greedy'],
                               beam_width=my_params['beam_width'], top_paths=my_params['top_paths'])[0][0],
                  dtype='float32')

class RecModel(object):
    def __init__(self, params):
        """
        Detection model for OCR text detection.
        :param params:
        """
        self.input_size = params['train']['image_shape'][1]
        self.greedy = params['train']['greedy']
        self.beam_width = params['train']['beam_width']
        self.top_paths = params['train']['top_paths']

        self.backbone = ResNet(params)
        self.head = SequenceEncoder(params)

    def __call__(self):

        image_input = layers.Input(shape=(self.input_size, self.input_size, 3))

        conv_feas = self.backbone(image_input)
        y_pred = self.head(conv_feas)

        """
        Transcription  就是把每个width 的预测 （per-frame prediction） 变成标记序列。
        这里边有两个方法：
        1. 没有词库的方法
        2. 有词库的方法
        
        没有词库的话，这个过程就是 Connectionist Temporal Classification (CTC) Loss 计算的过程。
        有词库呢，就会把预测的结果和词库中相似的词一一计算概率， 然后选择概率最大的那一个。 但是这个方法实在太简单粗暴了，
        当词库的单词很多的时候，就会花很多时间来计算概率。 而paper作者发现，可以先用没有词库的方法， 也就是CTC loss 算出一个sequence label,  
        然后寻找最相近的方法来确定单词 （search the nearest neighbour hood method）。
       """

        # Change shape
        labels = layers.Input(name='the_labels', shape=[None, ], dtype='int32')
        input_length = layers.Input(name='input_length', shape=[1], dtype='int32')
        label_length = layers.Input(name='label_length', shape=[1], dtype='int32')

        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='CTCloss')([y_pred,
                                                                           labels,
                                                                           input_length,
                                                                           label_length])

        # Lambda layer for the decoding function
        out_decoded_dense = layers.Lambda(ctc_complete_decoding_lambda_func, output_shape=(None, None),
                                   name='CTCdecode', arguments={'greedy': self.greedy,
                                                                'beam_width': self.beam_width,
                                                                'top_paths': self.top_paths}, dtype="float32")(
            y_pred + [input_length])


        training_model = models.Model(inputs=[image_input, labels, input_length, label_length], outputs=loss_out)

        prediction_model = models.Model(inputs=image_input, outputs=y_pred)

        decode_model = models.Model(inputs=image_input + [input_length], outputs=out_decoded_dense)

        return training_model, prediction_model, decode_model

