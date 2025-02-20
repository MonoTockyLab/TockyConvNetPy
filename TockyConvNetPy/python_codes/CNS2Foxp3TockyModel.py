tf.keras.backend.clear_session()
from tensorflow.keras import regularizers

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, multiply, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation, GlobalAveragePooling2D, Reshape, Multiply, Layer, InputSpec
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
        
def spatial_attention_block(input_tensor):
    attention_map = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(input_tensor)
    output_tensor = Multiply()([input_tensor, attention_map])
    return output_tensor
    
input_tensor = Input(shape=(100, 100, 1))

# First Convolutional Layer
x = Conv2D(16, (3,3), padding='same')(input_tensor)
x = Activation('relu')(x)
x = spatial_attention_block(x)  # Apply attention
x = MaxPooling2D((2,2))(x)
x = Dropout(0.2)(x)

# Second Convolutional Layer with Squeeze-and-Excitation Block
x = Conv2D(16, (3,3), padding='same')(x)#, kernel_regularizer=regularizers.l2(0.01)
x = Activation('relu')(x)
x = spatial_attention_block(x)  # Apply attention
x = MaxPooling2D((2,2))(x)
x = Dropout(0.2)(x)

# Fully connected layers
x = Flatten()(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.5)(x)
# Output Layer
output_tensor = Dense(2, activation='softmax')(x)

model = Model(input_tensor, output_tensor)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
