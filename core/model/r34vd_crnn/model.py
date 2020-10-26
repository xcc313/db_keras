from  .backbone import ResNet
from .head import CTCHead
from .loss import DBLoss

from keras import layers, models


class RecModel(object):
    def __init__(self, params):
        """
        Detection model for OCR text detection.
        :param params:
        """
        self.input_size = params['train']['image_shape'][1]
        self.backbone = ResNet(params)
        self.head = CTCHead(params)
        self.loss = DBLoss(params)

    def __call__(self):

        image_input = layers.Input(shape=(self.input_size, self.input_size, 3))
        gt_input = layers.Input(shape=(self.input_size, self.input_size))
        mask_input = layers.Input(shape=(self.input_size, self.input_size))
        thresh_input = layers.Input(shape=(self.input_size, self.input_size))
        thresh_mask_input = layers.Input(shape=(self.input_size, self.input_size))

        conv_feas = self.backbone(image_input)
        shrink_maps, threshold_maps, binary_maps = self.head(conv_feas)
        losses = layers.Lambda(self.loss, name='loss_all')(
            [shrink_maps, threshold_maps, binary_maps, gt_input, mask_input, thresh_input, thresh_mask_input])

        training_model = models.Model(inputs=[image_input, gt_input, mask_input, thresh_input, thresh_mask_input],
                                      outputs=losses)
        prediction_model = models.Model(inputs=image_input, outputs=shrink_maps)

        return training_model, prediction_model

