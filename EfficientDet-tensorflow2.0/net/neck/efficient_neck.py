# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:efficient_neck.py
# software: PyCharm

import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPool2D, UpSampling2D, BatchNormalization
import tensorflow as tf


class Swish(keras.layers.Layer):
    """
        swish activation
        y = x * sigmoid(x)
    """
    def __init__(self):
        super(Swish, self).__init__(name='swish')

    def call(self, inputs, **kwargs):
        result = tf.nn.swish(inputs)
        return result


class WeightAdd(keras.layers.Layer):
    """
        weighted feature fusion
    """
    def __init__(self, epsilon=1e-4, **kwargs):
        super(WeightAdd, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_weights = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_weights,),
                                 dtype=tf.float32,
                                 trainable=True,
                                 initializer=keras.initializers.constant(1 / num_weights))

    def call(self, inputs, **kwargs):
        w = keras.activations.relu(self.w)
        w = w / (tf.reduce_sum(w) + self.epsilon)
        results = [inputs[i] * w[i] for i in range(len(inputs))]
        results = tf.reduce_sum(results, axis=0)
        return results


def bifpn(features, out_channels, ids):

    # bifpn的第一层
    if ids == 0:
        _, _, c3, c4, c5 = features
        p3 = Conv2D(out_channels, 1, 1, padding='same')(c3)
        p3 = BatchNormalization()(p3)
        # p3 = Swish()(p3)

        p4 = Conv2D(out_channels, 1, 1, padding='same')(c4)
        p4 = BatchNormalization()(p4)
        # p4 = Swish()(p4)

        p5 = Conv2D(out_channels, 1, 1, padding='same')(c5)
        p5 = BatchNormalization()(p5)
        # p5 = Swish()(p5)

        p6 = Conv2D(out_channels, 1, 1, padding='same')(c5)
        p6 = BatchNormalization()(p6)
        p6 = MaxPool2D(3, 2, padding='same')(p6)

        p7 = MaxPool2D(3, 2, padding='same')(p6)
        p7_up = UpSampling2D(2)(p7)

        p6_middle = WeightAdd()([p6, p7_up])
        p6_middle = Swish()(p6_middle)
        p6_middle = SeparableConv2D(out_channels, 3, 1, padding='same')(p6_middle)
        p6_middle = BatchNormalization()(p6_middle)
        p6_up = UpSampling2D(2)(p6_middle)

        p5_middle = WeightAdd()([p5, p6_up])
        p5_middle = Swish()(p5_middle)
        p5_middle = SeparableConv2D(out_channels, 3, padding='same')(p5_middle)
        p5_middle = BatchNormalization()(p5_middle)
        p5_up = UpSampling2D(2)(p5_middle)

        p4_middle = WeightAdd()([p4, p5_up])
        p4_middle = Swish()(p4_middle)
        p4_middle = SeparableConv2D(out_channels, 3, padding='same')(p4_middle)
        p4_middle = BatchNormalization()(p4_middle)
        p4_up = UpSampling2D(2)(p4_middle)

        p3_out = WeightAdd()([p3, p4_up])
        p3_out = Swish()(p3_out)
        p3_out = SeparableConv2D(out_channels, 3, padding='same')(p3_out)
        p3_out = BatchNormalization()(p3_out)
        p3_down = MaxPool2D(3, strides=2, padding='same')(p3_out)

        # path aggregation
        p4_out = WeightAdd()([p4, p4_middle, p3_down])
        p4_out = Swish()(p4_out)
        p4_out = SeparableConv2D(out_channels, kernel_size=3, strides=1, padding='same')(p4_out)
        p4_out = BatchNormalization()(p4_out)
        p4_down = MaxPool2D(pool_size=3, strides=2, padding='same')(p4_out)

        p5_out = WeightAdd()([p5, p5_middle, p4_down])
        p5_out = Swish()(p5_out)
        p5_out = SeparableConv2D(out_channels, 3, 1, padding='same')(p5_out)
        p5_out = BatchNormalization()(p5_out)
        p5_down = MaxPool2D(3, strides=2, padding='same')(p5_out)

        p6_out = WeightAdd()([p6, p6_middle, p5_down])
        p6_out = Swish()(p6_out)
        p6_out = SeparableConv2D(out_channels, kernel_size=3, strides=1, padding='same')(p6_out)
        p6_out = BatchNormalization()(p6_out)
        p6_down = MaxPool2D(pool_size=3, strides=2, padding='same')(p6_out)

        p7_out = WeightAdd()([p7, p6_down])
        p7_out = Swish()(p7_out)
        p7_out = SeparableConv2D(out_channels, kernel_size=3, strides=1, padding='same')(p7_out)
        p7_out = BatchNormalization()(p7_out)

    # bifpn其他层
    else:
        p3, p4, p5, p6, p7 = features

        p7_up = UpSampling2D(2)(p7)

        p6_middle = WeightAdd()([p6, p7_up])
        p6_middle = Swish()(p6_middle)
        p6_middle = SeparableConv2D(out_channels, 3, 1, padding='same')(p6_middle)
        p6_middle = BatchNormalization()(p6_middle)
        p6_up = UpSampling2D(2)(p6_middle)

        p5_middle = WeightAdd()([p5, p6_up])
        p5_middle = Swish()(p5_middle)
        p5_middle = SeparableConv2D(out_channels, 3, padding='same')(p5_middle)
        p5_middle = BatchNormalization()(p5_middle)
        p5_up = UpSampling2D(2)(p5_middle)

        p4_middle = WeightAdd()([p4, p5_up])
        p4_middle = Swish()(p4_middle)
        p4_middle = SeparableConv2D(out_channels, 3, padding='same')(p4_middle)
        p4_middle = BatchNormalization()(p4_middle)
        p4_up = UpSampling2D(2)(p4_middle)

        p3_out = WeightAdd()([p3, p4_up])
        p3_out = Swish()(p3_out)
        p3_out = SeparableConv2D(out_channels, 3, padding='same')(p3_out)
        p3_out = BatchNormalization()(p3_out)
        p3_down = MaxPool2D(3, strides=2, padding='same')(p3_out)

        # path aggregation
        p4_out = WeightAdd()([p4, p4_middle, p3_down])
        p4_out = Swish()(p4_out)
        p4_out = SeparableConv2D(out_channels, kernel_size=3, strides=1, padding='same')(p4_out)
        p4_out = BatchNormalization()(p4_out)
        p4_down = MaxPool2D(pool_size=3, strides=2, padding='same')(p4_out)

        p5_out = WeightAdd()([p5, p5_middle, p4_down])
        p5_out = Swish()(p5_out)
        p5_out = SeparableConv2D(out_channels, 3, 1, padding='same')(p5_out)
        p5_out = BatchNormalization()(p5_out)
        p5_down = MaxPool2D(3, strides=2, padding='same')(p5_out)

        p6_out = WeightAdd()([p6, p6_middle, p5_down])
        p6_out = Swish()(p6_out)
        p6_out = SeparableConv2D(out_channels, kernel_size=3, strides=1, padding='same')(p6_out)
        p6_out = BatchNormalization()(p6_out)
        p6_down = MaxPool2D(pool_size=3, strides=2, padding='same')(p6_out)

        p7_out = WeightAdd()([p7, p6_down])
        p7_out = Swish()(p7_out)
        p7_out = SeparableConv2D(out_channels, kernel_size=3, strides=1, padding='same')(p7_out)
        p7_out = BatchNormalization()(p7_out)

    return p3_out, p4_out, p5_out, p6_out, p7_out


class ClassPrediction(keras.Model):
    """
        (h, w, num_anchors, num_classes)
        y_true的制作和faster-rcnn一致，one gt map many anchors
        the anchor assign in yolo is greedy, one anchor response one gt
    """
    # TODO: complete class prediction model
    pass


class BoxPrediction(keras.Model):
    """
        (h, w, num_anchors, 4)
    """
    # TODO: complete box prediction model
    pass


























if __name__ == '__main__':
    # inputs = keras.Input(shape=(224, 224, 3))
    # res = Swish()(inputs)
    # model = keras.Model(inputs, res)
    # model(inputs, training=False)
    # model.summary()















