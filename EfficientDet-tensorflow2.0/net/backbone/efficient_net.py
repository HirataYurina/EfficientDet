# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:efficient_net.py
# software: PyCharm

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import math as math


# MBBlock params * 7
mbblock_params = [{'kernel_size': 3, 'strides': 1, 'input_filters': 32, 'out_filters': 16,
                   'num_layers': 1, 'expand_ratio': 1, 'se_ratio': 0.25},
                  {'kernel_size': 3, 'strides': 2, 'input_filters': 16, 'out_filters': 24,
                   'num_layers': 2, 'expand_ratio': 6, 'se_ratio': 0.25},
                  {'kernel_size': 5, 'strides': 2, 'input_filters': 24, 'out_filters': 40,
                   'num_layers': 2, 'expand_ratio': 6, 'se_ratio': 0.25},
                  {'kernel_size': 3, 'strides': 2, 'input_filters': 40, 'out_filters': 80,
                   'num_layers': 3, 'expand_ratio': 6, 'se_ratio': 0.25},
                  {'kernel_size': 5, 'strides': 1, 'input_filters': 80, 'out_filters': 112,
                   'num_layers': 3, 'expand_ratio': 6, 'se_ratio': 0.25},
                  {'kernel_size': 5, 'strides': 2, 'input_filters': 112, 'out_filters': 192,
                   'num_layers': 4, 'expand_ratio': 6, 'se_ratio': 0.25},
                  {'kernel_size': 3, 'strides': 1, 'input_filters': 192, 'out_filters': 320,
                   'num_layers': 1, 'expand_ratio': 6, 'se_ratio': 0.25},
                  ]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}


def round_depth(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier"""
    return int(math.ceil(repeats * depth_coefficient))


def round_filters(filters, width_coefficient, depth_divisor):
    """

    Args:
        filters: the width of this layer
        width_coefficient: the scale coefficient
        depth_divisor: filter的数量需要为8的整数倍，40~43->40，43~48->48

    Returns: new_filters

    """
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def mb_block(inputs, mb_block_aug):
    # mb_block_aug
    has_se = (mb_block_aug['se_ratio'] is not None and (0 < mb_block_aug['se_ratio'] <= 1))

    expand_ratio = mb_block_aug['expand_ratio']

    if mb_block_aug['expand_ratio'] != 1:
        x = layers.Conv2D(mb_block_aug['input_filters'] * expand_ratio, 1, padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        # use tf.float32, so just use relu
        x = layers.ReLU()(x)
    else:
        x = inputs

    x = layers.DepthwiseConv2D(mb_block_aug['kernel_size'], strides=mb_block_aug['strides'],
                               padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # squeeze and excitation
    if has_se:
        num_se_filters = max(1, int(mb_block_aug['input_filters'] * expand_ratio * mb_block_aug['se_ratio']))
        se_x = layers.GlobalAvgPool2D()(x)
        se_x = layers.Dense(num_se_filters, activation='relu')(se_x)
        se_x = layers.Dense(mb_block_aug['input_filters'] * expand_ratio, activation='sigmoid')(se_x)
        # (1, 1, num_filters)
        se_x = layers.Reshape(target_shape=(1, 1, mb_block_aug['input_filters'] * expand_ratio))(se_x)
        x = layers.Multiply()([x, se_x])

    # linear activation
    x = layers.Conv2D(mb_block_aug['out_filters'], 1, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if mb_block_aug['strides'] == 1 and mb_block_aug['input_filters'] == mb_block_aug['out_filters']:
        # skip connect
        x = layers.Add()([inputs, x])

    return x


def efficient_net(default_resolution,
                  width_ratio,
                  depth_ratio,
                  block_args,
                  depth_divisor=8):
    """
    efficient net with scale coefficient
    Args:
        depth_divisor: int
        default_resolution: default input shape
        width_ratio: width scale coefficient
        depth_ratio: depth scale coefficient
        block_args: baseline block args

    Returns: efficient outputs [c1, c2, c3, c4, c5]

    """
    inputs = keras.Input(shape=(default_resolution, default_resolution, 3))

    x = layers.Conv2D(round_filters(32, width_ratio, depth_divisor), 3, 2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    outputs = []

    for index, block_arg in enumerate(block_args):
        # update block_args based on scale coefficient
        block_arg.update({'input_filters': round_filters(block_arg['input_filters'], width_ratio, depth_divisor),
                          'out_filters': round_filters(block_arg['out_filters'], width_ratio, depth_divisor),
                          'num_layers': round_depth(block_arg['num_layers'], depth_ratio)})
        x = mb_block(x, block_arg)

        # if num_layers > 1, continue
        # update 'input_filters' and 'strides'
        if block_arg['num_layers'] > 1:
            block_arg.update({'input_filters': block_arg['out_filters'],
                              'strides': 1})
            for i in range(1, block_arg['num_layers']):
                x = mb_block(x, block_arg)

        if index in [0, 1, 2, 4, 6]:
            outputs.append(x)

    return outputs, inputs


def efficient_net_b0():
    return efficient_net(224, 1.0, 1.0, mbblock_params)


def efficient_net_b1():
    return efficient_net(240, 1.0, 1.1, mbblock_params)


def efficient_net_b2():
    return efficient_net(260, 1.1, 1.2, mbblock_params)


def efficient_net_b3():
    return efficient_net(300, 1.2, 1.4, mbblock_params)


def efficient_net_b4():
    return efficient_net(380, 1.4, 1.8, mbblock_params)


def efficient_net_b5():
    return efficient_net(456, 1.6, 2.2, mbblock_params)


def efficient_net_b6():
    return efficient_net(528, 1.8, 2.6, mbblock_params)


def efficient_net_b7():
    return efficient_net(600, 2.0, 3.1, mbblock_params)


if __name__ == '__main__':
    # test efficient net
    features, input_tensor = efficient_net_b0()
    eff_model = keras.Model(inputs=input_tensor, outputs=features)
    # eff_model.summary()
    for feature in features:
        print(feature.shape)
