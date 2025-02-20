# model_module.py

# Imports
import os
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, Multiply
from tensorflow.keras.optimizers import Adam


def spatial_attention_block(input_tensor):
    """
    Applies a spatial attention mechanism on the input tensor.
    Args:
    input_tensor (Tensor): Input tensor for the attention block.

    Returns:
    Tensor: Output tensor after applying spatial attention.
    """
    attention_map = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(input_tensor)
    output_tensor = Multiply()([input_tensor, attention_map])
    return output_tensor

def build_model():
    
    # Input layer
    input_tensor = Input(shape=(100, 100, 1))

    # First Convolutional Layer
    x = Conv2D(16, (3, 3), padding='same')(input_tensor)
    x = Activation('relu')(x)
    x = spatial_attention_block(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    # Second Convolutional Layer
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = spatial_attention_block(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    # Fully Connected Layers
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Output Layer
    output_tensor = Dense(2, activation='softmax')(x)

    # Model assembly
    model = Model(input_tensor, output_tensor)

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    # If run as a script, create and show the model summary.
    model = build_model()
    model.summary()

