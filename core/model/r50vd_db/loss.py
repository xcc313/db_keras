
import numpy as np
import tensorflow as tf

def BalanceLoss(pred,
                gt,
                mask,
                balance_loss=True,
                main_loss_type="DiceLoss",
                negative_ratio=3,
                return_origin=False,
                eps=1e-6):
    """
    The BalanceLoss for Differentiable Binarization text detection
    args:
        pred (variable): predicted feature maps.
        gt (variable): ground truth feature maps.
        mask (variable): masked maps.
        balance_loss (bool): whether balance loss or not, default is True
        main_loss_type (str): can only be one of ['CrossEntropy','DiceLoss',
            'Euclidean','BCELoss', 'MaskL1Loss'], default is  'DiceLoss'.
        negative_ratio (int|float): float, default is 3.
        return_origin (bool): whether return unbalanced loss or not, default is False.
        eps (float): default is 1e-6.
    return: (variable) balanced loss
    """
    positive = gt * mask
    negative = (1 - gt) * mask

    positive_count = tf.reduce_sum(positive)
    positive_count_int = tf.cast(positive_count, dtype=np.int32)
    negative_count = tf.reduce_min([
        tf.reduce_sum(negative), positive_count * negative_ratio])
    negative_count_int = tf.cast(negative_count, dtype=np.int32)

    if main_loss_type == "CrossEntropy":
        loss = tf.cross_entropy(input=pred, label=gt, soft_label=True)
        loss = tf.reduce_mean(loss)
    elif main_loss_type == "Euclidean":
        loss = tf.square(pred - gt)
        loss = tf.reduce_mean(loss)
    elif main_loss_type == "DiceLoss":
        loss = DiceLoss(pred, gt, mask)
    elif main_loss_type == "BCELoss":
        loss = tf.sigmoid_cross_entropy_with_logits(pred, label=gt)
    elif main_loss_type == "MaskL1Loss":
        loss = MaskL1Loss(pred, gt, mask)
    else:
        loss_type = [
            'CrossEntropy', 'DiceLoss', 'Euclidean', 'BCELoss', 'MaskL1Loss'
        ]
        raise Exception("main_loss_type in BalanceLoss() can only be one of {}".
                        format(loss_type))

    if not balance_loss:
        return loss

    positive_loss = positive * loss
    negative_loss = negative * loss
    negative_loss = tf.reshape(negative_loss, shape=[-1])
    negative_loss, _ = tf.nn.top_k(negative_loss, k=negative_count_int)
    balance_loss = (tf.reduce_sum(positive_loss) +
                    tf.reduce_sum(negative_loss)) / (
                        positive_count + negative_count + eps)

    if return_origin:
        return balance_loss, loss
    return balance_loss


def DiceLoss(pred, gt, mask, weights=None, eps=1e-6):
    """
    DiceLoss function.
    """
    if weights is not None:
        mask = weights * mask
    intersection = tf.reduce_sum(pred * gt * mask)

    union = tf.reduce_sum(pred * mask) + tf.reduce_sum(
        gt * mask) + eps
    loss = 1 - 2.0 * intersection / union
    return loss


def MaskL1Loss(pred, gt, mask, eps=1e-6):
    """
    Mask L1 Loss
    """
    loss = tf.reduce_sum((tf.abs(pred - gt) * mask)) / (
        tf.reduce_sum(mask) + eps)
    loss = tf.reduce_mean(loss)
    return loss

class DBLoss(object):
    """
    Differentiable Binarization (DB) Loss Function
    args:
        param (dict): the super paramter for DB Loss
    """

    def __init__(self, params):
        super(DBLoss, self).__init__()
        self.__name__ = 'DBLoss'
        self.main_loss_type = params['det']['main_loss_type']
        self.alpha = params['det']['alpha']
        self.beta = params['det']['beta']


    def __call__(self, arg):
        shrink_maps, threshold_maps, binary_maps, label_shrink_map, label_shrink_mask, label_threshold_map, label_threshold_mask  = arg
        shrink_maps=shrink_maps[..., 0]
        threshold_maps=threshold_maps[..., 0]
        binary_maps=binary_maps[..., 0]

        loss_shrink_maps = BalanceLoss(
            shrink_maps,
            label_shrink_map,
            label_shrink_mask,
            main_loss_type=self.main_loss_type)
        loss_threshold_maps = MaskL1Loss(threshold_maps, label_threshold_map,
                                         label_threshold_mask)
        loss_binary_maps = DiceLoss(binary_maps, label_shrink_map,
                                    label_shrink_mask)
        loss_shrink_maps = self.alpha * loss_shrink_maps
        loss_threshold_maps = self.beta * loss_threshold_maps

        loss_all = loss_shrink_maps + loss_threshold_maps\
            + loss_binary_maps

        return loss_all
