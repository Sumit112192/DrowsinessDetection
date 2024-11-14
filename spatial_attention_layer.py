import tensorflow as tf
import keras

@keras.utils.register_keras_serializable()
class SpatialAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialAttentionLayer, self).__init__(**kwargs)
        # Create the Conv2D layer outside the call method
        self.attention_conv = tf.keras.layers.Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, input_feature):
        avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_feature)
        max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_feature)
        concat = tf.concat([avg_pool, max_pool], axis=3)
        # Use the pre-created Conv2D layer
        attention = self.attention_conv(concat)
        output = input_feature * attention
        return output
        
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
