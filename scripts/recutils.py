from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf
import keras as ks


def compute_class_weights(y_true_multilabel):

    n_classes = np.shape(y_true_multilabel)[1]
    weights = np.ones([n_classes, 2])
    for c in range(0, n_classes):
        if np.max(y_true_multilabel[:, c]) > 0:
            weights[c] = compute_class_weight('balanced', [0, 1], y_true_multilabel[:, c])
    return np.transpose(weights)


def focal_loss(gamma=2.0, alpha=0.25):
    '''
    Focal loss for binary classification
    :param gamma: focusing parameter of the focal loss factors
    :param alpha: balancing parameter of the focal loss factors
    :return: focal loss function
    '''
    def _focal_loss(y_true, y_pred):
        eps = ks.backend.epsilon()
        y_pred = ks.backend.clip(y_pred, eps, 1. - eps)
        s_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        s_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        # w_0 = ks.backend.variable(weights[0, :])
        # w_1 = ks.backend.variable(weights[1, :])
        # return -ks.backend.sum(w_1 * ks.backend.pow(1. - s_1, gamma) * ks.backend.log(s_1)) \
        #        -ks.backend.sum(w_0 * ks.backend.pow(s_0, gamma) * ks.backend.log(1. - s_0))
        return -ks.backend.sum(alpha * ks.backend.pow(1. - s_1, gamma) * ks.backend.log(s_1)) \
               -ks.backend.sum((1 - alpha) * ks.backend.pow(s_0, gamma) * ks.backend.log(1. - s_0))

    return _focal_loss


def weighted_loss(weights):
    '''
    Weighted loss function
    :param weights: binary weights to apply to each basket item
    :return: weighted loss function
    '''
    def _weighted_loss(y_true, y_pred):
        return ks.backend.mean((weights[0, :]**(1-y_true))*(weights[1, :]**y_true)*ks.backend.binary_crossentropy(y_true, y_pred), axis=-1)
    return _weighted_loss


def multibinlabel_acc_1d(y_true, y_pred):
    y_true = np.argwhere(y_true).ravel()
    y_pred = np.argpartition(y_pred, -y_true.size)[-y_true.size:]
    matches = np.intersect1d(y_pred, y_true).size
    return matches / y_true.size


def multilabel_acc(y_true, y_pred):
    matches = 0
    y_pred = np.argpartition(y_pred, -y_true.shape[1], axis=1)[:, -y_true.shape[1]:]
    for predictions, targets in zip(y_pred, y_true):
        matches += np.intersect1d(predictions, targets).size

    return matches / y_true.size


class Metrics(ks.callbacks.Callback):
    '''
    Metrics class extends Callback in order to add multilabel accuracy into the reported metrics
    '''

    def on_train_begin(self, logs={}):
        self.val_multilabel_acc = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_score = np.asarray(self.model.predict(self.validation_data[0]))
        val_targ = self.validation_data[1]

        _val_multilabel_acc = self.get_multilabel_acc(val_targ, val_score)

        self.val_multilabel_acc.append(_val_multilabel_acc)
        logs['val_multilabel_acc'] = _val_multilabel_acc
        print(' - val_multilabel_acc: {:.4f}'.format(_val_multilabel_acc))
        return

    def get_multilabel_acc(self, y_true, y_pred):
        matches = 0
        total_count = 0
        for preds, targets in zip(y_pred, y_true):
            tar = np.argwhere(targets).ravel()
            pred = np.argpartition(preds, -tar.size)[-tar.size:]
            matches += np.intersect1d(pred, tar).size
            total_count += tar.size

        return matches / float(total_count)

