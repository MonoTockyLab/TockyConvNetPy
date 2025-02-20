# TockyConvNetPy/instance_normalization.py

import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="TockyConvNetPy", name="InstanceNormalization")
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = 1e-5

    def build(self, input_shape):
        depth = input_shape[-1]
        self.scale = self.add_weight(name='scale', shape=(depth,),
                                     initializer='ones', trainable=True)
        self.shift = self.add_weight(name='shift', shape=(depth,),
                                     initializer='zeros', trainable=True)
        super(InstanceNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (inputs - mean) * inv
        return self.scale * normalized + self.shift
