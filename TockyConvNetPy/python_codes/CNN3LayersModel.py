#!/usr/bin/env python
# coding: utf-8

# # Analysis Using TockyCNN 3 Conv Layers Model
# 
# ## Author: Dr Masahiro Ono  
# ## Date: 2025-02-16
# 
# ## Aim:
# This notebook demonstrates the complete analysis workflow using the TockyCNN model with 3 convolutional layers. The analysis includes loading a pre-trained model, processing independent test data, evaluating model performance (with ROC and Precision-Recall curves), performing Grad-CAM analysis across multiple convolutional layers, and conducting regression analyses on continuous score outputs. All results are saved for further review.
# 

# ---
# ## 1. Importing Libraries and Setting Up the Environment
# 
# In this cell, we import all necessary libraries and modules. These include standard packages for data manipulation (NumPy, pandas), visualization (Matplotlib, seaborn), deep learning (TensorFlow/Keras), statistical modeling (statsmodels), and utilities from scikit-learn. Custom functions and objects (such as `InstanceNormalization`, `find_optimal_threshold`, `smooth_heatmap`, and `generate_aggregated_heatmap`) are imported from the `TockyConvNetPy` package.
# 

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import matplotlib as mpl
import importlib.resources

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, 
                             roc_curve, roc_auc_score, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from TockyConvNetPy import (InstanceNormalization, find_optimal_threshold, 
                            smooth_heatmap, generate_aggregated_heatmap)



# ---
# ## 2. Loading the Pre-trained 3-Conv Layer Model
# 
# Here we load the pre-trained 3-convolutional-layer model using Keras’ `load_model` function. The custom layer `InstanceNormalization` is passed to ensure correct model loading. Finally, we print the model summary to verify the architecture.
# 

# In[2]:


#Load Model
custom_objects = {"InstanceNormalization": InstanceNormalization}

with importlib.resources.path("TockyConvNetPy.data.Foxp3DevAge.models", "Foxp3DevAgeCNN3Layers.keras") as model_path:
    model = load_model(str(model_path),custom_objects=custom_objects)

model.summary()


# ## 3. Importing and Preprocessing Independent Test Data
# 
# This cell loads independent test data, including images, labels, age, and timer-positive data. The age and timer data are preprocessed (for example, timer data is scaled and age data is used with the custom `log_scale_age` function). We also create new labels by combining organ type and age group, encode these labels, and compute class indices. This step sets up the data for both performance evaluation and subsequent Grad-CAM analysis.
# 

# In[3]:


#Import Independent Test Data
all_images = np.load('all_test_data/sample_images.npy')
all_labels = np.load('all_test_data/sample_labels.npy')
age_data = pd.read_csv('all_test_data/sampledef_age.csv')
timer_pos_data = pd.read_csv('all_test_data/timer_pos.csv')
timer_pos_data = timer_pos_data['timer_positive'].values
all_ages = age_data['age'].values
all_timers_scaled = timer_pos_data/100

new_labels = []
for i in range(len(all_labels)):
    organ = "Spleen" if all_labels[i,0] == 1 else "Thymus"
    age_group = "Aged" if all_ages[i] >= 30 else "Young"
    new_labels.append(organ + age_group)

new_labels = np.array(new_labels)

encoder = LabelEncoder()
integer_labels = encoder.fit_transform(new_labels)
one_hot_labels = to_categorical(integer_labels, num_classes=4)
class_indices = {class_name: np.where(integer_labels == i)[0] for i, class_name in enumerate(encoder.classes_)}

print("Number of samples in each class:")
for class_name, indices in class_indices.items():
    print(f"{class_name}: {len(indices)} samples")


# ---
# ## 4. Evaluating Model Performance
# 
# In this section, we compute performance metrics for the classifier. The model predicts class probabilities on the test set, and an optimal threshold is determined for each class (using ROC analysis) to derive final predicted labels. Overall and per-class confusion matrices are printed, and ROC and Precision-Recall curves are generated and visualized. The results (metrics and curves) are saved to CSV files and exported as a PDF.
# 

# In[4]:


# Obtain Model Performance Metrics

base_export_dir = 'CNN3LayersModelPerformance'
os.makedirs(base_export_dir, exist_ok=True)

predictions = model.predict([all_images, all_timers_scaled])
encoder = LabelEncoder()
true_classes = encoder.fit_transform(new_labels)
classes = encoder.classes_
predicted_classes = np.zeros(predictions.shape[0], dtype=int)
for i in range(predictions.shape[1]):
    fpr, tpr, thresholds = roc_curve((true_classes == i).astype(int), predictions[:, i])
    optimal_threshold = find_optimal_threshold(fpr, tpr, thresholds)
    predicted_classes[predictions[:, i] >= optimal_threshold] = i

cm_overall = confusion_matrix(true_classes, predicted_classes)
print("Overall Confusion Matrix:")
print(cm_overall)

fig, axes = plt.subplots(2, 4, figsize=(24, 12))
roc_axes = axes[0, :]
pr_axes = axes[1, :]
axis_label_font_size = 22
title_font_size = 24 
tick_label_size = 18 

for i, ax in enumerate(roc_axes.ravel()):
    class_name = classes[i]
    true_binary = (true_classes == i)
    predicted_binary = (predicted_classes == i)
    class_dir = os.path.join(base_export_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    cm_class = confusion_matrix(true_binary, predicted_binary)
    print(f"Confusion Matrix for {class_name}:")
    print(cm_class)

    if len(np.unique(true_binary)) > 1: 
        fpr, tpr, roc_thresholds = roc_curve(true_binary, predictions[:, i])
        roc_auc = roc_auc_score(true_binary, predictions[:, i])
        precision_values, recall, pr_thresholds = precision_recall_curve(true_binary, predictions[:, i])
        average_precision = average_precision_score(true_binary, predictions[:, i])
        positive_class_probabilities = predictions[:, i]

        # ROC Curves
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=axis_label_font_size)
        ax.set_ylabel('True Positive Rate', fontsize=axis_label_font_size)
        ax.set_title(f'ROC: {class_name}', fontsize=title_font_size)
        ax.legend(loc="lower right", fontsize=20)
        ax.tick_params(axis='both', labelsize=tick_label_size)

        # PR Curves
        pr_ax = pr_axes[i]
        pr_ax.plot(recall, precision_values, color='blue', lw=2, label=f'Average Precision = {average_precision:.2f}')
        pr_ax.set_xlabel('Recall', fontsize=axis_label_font_size)
        pr_ax.set_ylabel('Precision', fontsize=axis_label_font_size)
        pr_ax.set_xlim([0.0, 1.0])
        pr_ax.set_ylim([0.0, 1.05])
        pr_ax.set_title(f'Precision-Recall: {class_name}', fontsize=title_font_size)
        pr_ax.legend(loc="lower right", fontsize=20)
        pr_ax.tick_params(axis='both', labelsize=tick_label_size)

        np.savetxt(os.path.join(class_dir, 'fpr.csv'), fpr, delimiter=',', header='fpr', comments='')
        np.savetxt(os.path.join(class_dir, 'tpr.csv'), tpr, delimiter=',', header='tpr', comments='')
        np.savetxt(os.path.join(class_dir, 'thresholds.csv'), thresholds, delimiter=',', header='thresholds', comments='')
        np.savetxt(os.path.join(class_dir, 'positive_class_probabilities.csv'), positive_class_probabilities, delimiter=',', header='probability', comments='')
        np.savetxt(os.path.join(class_dir, 'true_classes.csv'), true_binary, delimiter=',', header='true_class', comments='')
        np.savetxt(os.path.join(class_dir, 'precision.csv'), precision_values, delimiter=',', header='precision', comments='')
        np.savetxt(os.path.join(class_dir, 'recall.csv'), recall, delimiter=',', header='recall', comments='')

plt.tight_layout()
plt.show()

# Save to PDF
pdf_path = os.path.join(base_export_dir, 'roc_pr_curves.pdf')
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved ROC and PR curves to {pdf_path}")


# ---
# ## 5. Grad-CAM Analysis Across Convolutional Layers
# 
# This cell performs Grad-CAM analysis on the pre-trained model using all available convolutional and attention layers. For each layer and for each class, the Grad-CAM heatmap is computed, smoothed, and then visualized. Additionally, each heatmap is saved as a CSV file. This detailed analysis helps identify which regions in the input images contribute most to the model's predictions.
# 

# In[5]:


# Grad-CAM Analysis Using Each of Convolutional Layers Available


layers = ['conv1', 'attention1_conv', 'conv2', 'attention2_conv', 'conv3', 'attention3_conv']
num_classes = len(next(iter(class_indices.values()))) 

pdf_path = os.path.join(base_export_dir, 'GradCAM.pdf')
with PdfPages(pdf_path) as pdf:
    fig, axs = plt.subplots(6, 4, figsize=(6, 9))
    cmap = mpl.colormaps['viridis']

    for layer_idx, layer_name in enumerate(layers):
        print(f"Grad-CAM Analysis of {layer_name}")

        for i, (class_name, indices) in enumerate(class_indices.items()):
            ax = axs[layer_idx, i] 
            class_images = all_images[indices]
            class_timerpos = all_timers_scaled[indices]

            aggregated_heatmap = generate_aggregated_heatmap(class_images, model, layer_name, class_timerpos)
            smoothed_heatmap = smooth_heatmap(aggregated_heatmap, sigma = 5)

            image = ax.imshow(smoothed_heatmap, cmap=cmap, origin='lower')
            ax.set_title(f"{class_name}_{layer_name}", fontsize=5)
            ax.tick_params(axis='both', which='major', pad=0.8, labelsize=4)
            ax.grid(True,  color='gray', linewidth=0.5, alpha=0.5)  

            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=4, pad=0.8)
            heatmap_df = pd.DataFrame(smoothed_heatmap)
            export_path = os.path.join(base_export_dir, f"{layer_name}_{class_name}_heatmap.csv")
            heatmap_df.to_csv(export_path, index=False)

    plt.tight_layout()
    plt.show()

    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved all Grad-CAM analysis to {pdf_path}")



