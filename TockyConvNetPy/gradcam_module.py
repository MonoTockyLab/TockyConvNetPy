# gradcam_module.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter, zoom
from tensorflow.keras.models import load_model

def make_gradcam_heatmap(img_array, model, conv_layer_name, timerpos_array = None, optimal_threshold=None, pred_index=1):

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(conv_layer_name).output, model.output]
    )
    
    if timerpos_array is not None:
        inputs_to_model = [img_array, timerpos_array]
    else:
        inputs_to_model = img_array


    with tf.GradientTape() as tape:
        conv_layer_output, preds = grad_model(inputs_to_model)
        prob_class_of_interest = preds[0][pred_index]
        
        if optimal_threshold is not None:
            if prob_class_of_interest >= optimal_threshold:
                focus_index = pred_index
            else:
                focus_index = 1 - pred_index
        else:
                    focus_index = pred_index
                    
        class_channel = preds[:, focus_index]

    grads = tape.gradient(class_channel, conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_layer_output = conv_layer_output[0]
    heatmap = conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-10)
    return heatmap.numpy()
    
def generate_aggregated_heatmap(images, model, conv_layer_name, timerpositives, threshold_value=0):

    aggregated_heatmap = np.zeros_like(images[0, :, :, 0])
    epsilon = 1e-10
    
    for img, timerpos in zip(images, timerpositives):
        img_array = np.expand_dims(img, axis=0)
       
        timerpos_array = np.expand_dims(timerpos, axis = 0).reshape(1,-1)
        preds = model.predict([img_array, timerpos_array], verbose=0)
        predicted_class = np.argmax(preds[0])
        
        heatmap = make_gradcam_heatmap(img_array, model, conv_layer_name, timerpos_array, pred_index=predicted_class)
        
        scale_factor = (img.shape[0] / heatmap.shape[0], img.shape[1] / heatmap.shape[1])
        heatmap_upscaled = zoom(heatmap, scale_factor, order=1)
        heatmap_sum = np.sum(heatmap_upscaled)
        if heatmap_sum == 0:
            heatmap_normalized = np.zeros_like(heatmap_upscaled)
        else:
            heatmap_normalized = heatmap_upscaled / (heatmap_sum + epsilon)

        heatmap_normalized[heatmap_normalized < 0] = 0
        aggregated_heatmap += heatmap_normalized
    aggregated_heatmap /= np.max(aggregated_heatmap + epsilon)

    return aggregated_heatmap
    

def smooth_heatmap(heatmap, sigma=2):

    smoothed_heatmap = gaussian_filter(heatmap, sigma=sigma)

    mean_val = np.mean(smoothed_heatmap)
    std_val = np.std(smoothed_heatmap)

    if std_val > 1e-3:
        normalized_heatmap = (smoothed_heatmap - mean_val) / std_val
    else:
        normalized_heatmap = smoothed_heatmap - mean_val
    
    return normalized_heatmap

def generate_density(images):

    density = np.mean(images, axis=0)
    return density / np.sum(density)

def visualize_heatmaps(aggregated_heatmap, smoothed_heatmap, difference_heatmap, title_suffix):

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    cmap = plt.get_cmap('viridis')
    diff_cmap = plt.get_cmap('coolwarm')

    # Display the aggregated heatmap
    ax[0].imshow(aggregated_heatmap, cmap=cmap, origin='lower')
    ax[0].set_title(f"Raw GradCAM {title_suffix}")
    ax[0].axis('off')

    # Display the smoothed heatmap
    ax[1].imshow(smoothed_heatmap, cmap=cmap, origin='lower')
    ax[1].set_title(f"Smoothed GradCAM {title_suffix}")
    ax[1].axis('off')

    # Display the difference heatmap
    ax[2].imshow(difference_heatmap, cmap=diff_cmap, origin='lower')
    ax[2].set_title(f"Difference Heatmap {title_suffix}")
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()


def find_optimal_threshold(fpr, tpr, thresholds, factor=0.5):

    gmeans = np.sqrt(tpr * (1-fpr))
    index = np.argmax(gmeans)
    return thresholds[index]

