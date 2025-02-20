# Foxp3DevAge Model Construction

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense,
                                     Activation, Multiply, Concatenate)
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from TockyConvNetPy import InstanceNormalization, spatial_attention_block

###########################################################################
#Conv 2 Layers 2-Class Classifier for CNS2 KO Foxp3 Tocky
    
image_input = Input(shape=(100, 100, 1))

# First Convolutional Layer
x = Conv2D(16, (3,3), padding='same')(image_input)
x = Activation('relu')(x)
x = spatial_attention_block(x)  # Apply attention
x = MaxPooling2D((2,2))(x)
x = Dropout(0.2)(x)

# Second Convolutional Layer
x = Conv2D(16, (3,3), padding='same')(x)
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

###########################################################################
# Conv 2 Layers 2-Class Classifier for Continuous Score Model

# Input layers
image_input = Input(shape=(100, 100, 1), name='image_input')
age_input = Input(shape=(1,), name='age_input')
timer_pos_input = Input(shape=(1,), name='timer_pos_input')

# First Convolutional Layer
x = Conv2D(16, (3, 3), padding='same',  name='conv1')(image_input)
x = InstanceNormalization(name='inst_norm1')(x)
x = Activation('relu', name='activation1')(x)
x = spatial_attention_block(x, name_prefix='attention1')
x = MaxPooling2D((2, 2), name='max_pool1')(x)
x = Dropout(0.2, name='dropout1')(x)

# Second Convolutional Layer
x = Conv2D(32, (3, 3), padding='same',  name='conv2')(x)
x = Activation('relu', name='activation2')(x)
x = spatial_attention_block(x, name_prefix='attention2')
x = MaxPooling2D((2, 2), name='max_pool2')(x)
x = Dropout(0.25, name='dropout2')(x)

# Flatten and concatenate with age input
flattened_features = Flatten(name='flatten')(x)
combined = Concatenate(name='concat')([flattened_features, age_input, timer_pos_input])

# Fully connected layers
x = Dense(128, activation='relu', kernel_regularizer = l2(0.01), name='dense_fc')(combined)
x = Dropout(0.5, name='dropout_fc')(x)

# Output Layer
output_tensor = Dense(2, activation='softmax', name='output')(x)

model = Model(inputs=[image_input, age_input, timer_pos_input], outputs=output_tensor)

########################################################################################
#Conv 2 Layers 2-Class Classifier for CNS2 KO Foxp3 Tocky
    
image_input = Input(shape=(100, 100, 1))

# First Convolutional Layer
x = Conv2D(16, (3,3), padding='same')(image_input)
x = Activation('relu')(x)
x = spatial_attention_block(x)  # Apply attention
x = MaxPooling2D((2,2))(x)
x = Dropout(0.2)(x)

# Second Convolutional Layer
x = Conv2D(16, (3,3), padding='same')(x)
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

###########################################################################
#Conv 2 Layers 2-Class Classifier for CNS2 KO Foxp3 Tocky
    
image_input = Input(shape=(100, 100, 1))

# First Convolutional Layer
x = Conv2D(16, (3,3), padding='same')(image_input)
x = Activation('relu')(x)
x = spatial_attention_block(x)  # Apply attention
x = MaxPooling2D((2,2))(x)
x = Dropout(0.2)(x)

# Second Convolutional Layer
x = Conv2D(16, (3,3), padding='same')(x)
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

###########################################################################
#Conv 3 Layers 4-Class Classifier

# Input layers
image_input = Input(shape=(100, 100, 1), name='image_input')
timer_pos_input = Input(shape=(1,), name='timer_pos_input')

# First Convolutional Layer
x = Conv2D(16, (3, 3), padding='same',  name='conv1')(image_input)
x = InstanceNormalization(name='inst_norm1')(x)
x = Activation('relu', name='activation1')(x)
x = spatial_attention_block(x, name_prefix='attention1')
x = MaxPooling2D((2, 2), name='max_pool1')(x)
x = Dropout(0.2, name='dropout1')(x)

# Second Convolutional Layer
x = Conv2D(32, (3, 3),  padding='same',  name='conv2')(x)
x = Activation('relu', name='activation2')(x)
x = spatial_attention_block(x, name_prefix='attention2')
x = MaxPooling2D((2, 2), name='max_pool2')(x)
x = Dropout(0.2, name='dropout2')(x)

# Third Convolutional Layer
x = Conv2D(64, (3, 3),kernel_regularizer = l2(0.01), padding='same',  name='conv3')(x)
x = Activation('relu', name='activation3')(x)
x = spatial_attention_block(x, name_prefix='attention3')
x = MaxPooling2D((2, 2), name='max_pool3')(x)
x = Dropout(0.25, name='dropout3')(x)

flattened_features = Flatten(name='flatten')(x)
combined = Concatenate(name='concat')([flattened_features, timer_pos_input])
x = Dense(128, activation='relu', kernel_regularizer = l2(0.01), name='dense_fc')(combined)
x = Dropout(0.5, name='dropout_fc')(x)
output_tensor = Dense(4, activation='softmax', name='output')(x)

model = Model(inputs=[image_input, timer_pos_input], outputs=output_tensor)



