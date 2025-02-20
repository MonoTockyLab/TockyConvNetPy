import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Use legacy Adam optimizer recommended for M1/M2 Macs
from tensorflow.keras.optimizers.legacy import Adam

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

unique_classes = np.unique(integer_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=integer_labels)

class_weight_dict = {i: class_weights[i] for i in unique_classes}
print("Class Weights:", class_weight_dict)

age_bins = pd.qcut(all_ages, q=4, labels=False, duplicates='drop')  # creates quartile-based bins

fold_histories = []
strat_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

fold_histories = []

for train_ids, val_ids in strat_kfold.split(all_images, age_bins):
    print(f'Training on fold {train_ids, val_ids}...')


from tensorflow.keras import layers, models
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

for fold, (train_ids, val_ids) in enumerate(strat_kfold.split(all_images, age_bins)):
    print(f'Training on fold {fold+1}...')
 
    history = model.fit([all_images[train_ids], all_timers[train_ids]], all_labels[train_ids],
                        validation_data=([all_images[val_ids], all_timers[val_ids]], all_labels[val_ids]),
                        epochs=6, batch_size=4)
    
    fold_histories.append(history.history)
