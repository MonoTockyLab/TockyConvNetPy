import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def load_data(directory):
    """
    Loads training, validation, and test data from specified directory.
    
    Args:
    directory (str): Path to the directory containing the data files.
    
    Returns:
    tuple: Tuple containing numpy arrays for train, validation, and test images and labels.
    """
    train_images = np.load(f'{directory}/train_images.npy')
    train_labels = np.load(f'{directory}/train_labels.npy')
    valid_images = np.load(f'{directory}/valid_images.npy')
    valid_labels = np.load(f'{directory}/valid_labels.npy')
    test_images = np.load(f'{directory}/test_images.npy')
    test_labels = np.load(f'{directory}/test_labels.npy')
    print("Data loaded successfully.")
    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels

def compute_weights(labels):
    """
    Computes class weights from labels to handle class imbalance.
    
    Args:
    labels (numpy.array): Array of labels.
    
    Returns:
    dict: Dictionary of class weights.
    """
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels.ravel())
    return {i: weight for i, weight in enumerate(class_weights)}

def train_model(model, train_images, train_labels, valid_images, valid_labels, epochs=12, batch_size=8, n_splits=3):
    """
    Trains a model using K-fold cross-validation.
    
    Args:
    model (tf.keras.Model): The neural network model to train.
    train_images (numpy.array): Training images.
    train_labels (numpy.array): Training labels.
    valid_images (numpy.array): Validation images.
    valid_labels (numpy.array): Validation labels.
    epochs (int): Number of epochs per fold.
    batch_size (int): Batch size for training.
    n_splits (int): Number of folds for cross-validation.
    
    Returns:
    list: List of training histories for each fold.
    """
    all_images = np.concatenate([train_images, valid_images], axis=0)
    all_labels = np.concatenate([train_labels, valid_labels], axis=0)
    class_weights = compute_weights(train_labels)

    kfold = KFold(n_splits=n_splits, shuffle=True)
    fold_histories = []
    for fold, (train_ids, val_ids) in enumerate(kfold.split(all_images)):
        print(f'Training on fold {fold+1}...')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(all_images[train_ids], all_labels[train_ids],
                            validation_data=(all_images[val_ids], all_labels[val_ids]),
                            epochs=epochs, batch_size=batch_size,
                            class_weight=class_weights)
        fold_histories.append(history.history)

    return fold_histories
    
    
def plot_learning_curves(history_dict):
    """
    Plots learning curves for training and validation loss and accuracy from a history dictionary.

    Args:
    history_dict (dict): Dictionary containing 'loss', 'val_loss', 'accuracy', and 'val_accuracy' as keys.
    """
    # Check if the history_dict contains the required keys
    if 'loss' in history_dict and 'val_loss' in history_dict:
        # Plot loss
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history_dict['loss'], label='Training Loss')
        plt.plot(history_dict['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    if 'accuracy' in history_dict and 'val_accuracy' in history_dict:
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history_dict['accuracy'], label='Training Accuracy')
        plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage: Assume 'model' is loaded or defined elsewhere
    model = load_model('data/models/dCNS2_trained_model.keras')  # Modify as necessary
    data_dir = 'data/sample_data'
    train_images, train_labels, valid_images, valid_labels, test_images, test_labels = load_data(data_dir)
    histories = train_model(model, train_images, train_labels, valid_images, valid_labels)
    plot_learning_curves(histories[-1])
