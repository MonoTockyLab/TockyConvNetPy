#!/usr/bin/env python
# coding: utf-8

# # Grad-CAM and Continuous Score Model Notebook
# 
# ## Author: Dr Masahiro Ono
# ## Date: 2025-02-16
# 
# ## Aim:
# This notebook demonstrates how to construct and analyse a continuous score model by transferring learnt model weights of a pre-trained 2-class classifier to a new logit model. 
# 
# Steps:
# 
# 1. Set up: import a pre-trained 2-class classifier model and load independent test data
# 2. Continuous Score Model Construction
# 3. Score Analysis Using Quadratic Regression 
# 

# In[1]:


#Import Neccessary Libraries

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import importlib.resources
import TockyConvNetPy
from TockyConvNetPy import InstanceNormalization, log_scale_age


# In[2]:


# Load Trained 2-Class Classifier
custom_objects = {"InstanceNormalization": InstanceNormalization}

with importlib.resources.path("TockyConvNetPy.data.Foxp3DevAge.models", "Foxp3DevAge2ClassClassifier.keras") as model_path:
    model = load_model(str(model_path),custom_objects=custom_objects)

model.summary()


# ---
# # Construction of New Continuous Score Model
# 
# The following operations are performed here:
# 
# - Extracting the penultimate layer (`dense_fc`) from the original classifier.
# - Adding a new dense layer (initially with 2 units and linear activation) to produce a combined output.
# - Transferring weights from the corresponding layers of the original model.
# - Replacing the final output with a logits layer that reuses the original output layer’s weights.
# - Finally, printing the model summary to confirm the architecture

# In[3]:


#Construct A Continuous Score Model
dense_fc_layer = model.get_layer('dense_fc').output
new_output = Dense(2, activation= None, name='combined_output')(dense_fc_layer)
continuous_score_model = Model(inputs=model.input, outputs=new_output)

for layer in continuous_score_model.layers:
    if layer.name in [l.name for l in model.layers] and 'output' not in layer.name:
        try:
            original_layer = model.get_layer(layer.name)
            original_weights = original_layer.get_weights()

            continuous_score_model.get_layer(layer.name).set_weights(original_weights)

        except Exception as e:
            print(f"Could not transfer weights for layer: {layer.name}. Error: {e}")

penultimate_output = model.get_layer('dense_fc').output

original_output_layer = model.get_layer('output')  #the output layer
logits_output = tf.keras.layers.Dense(
    original_output_layer.units,  
    activation=None, 
    name='logits_output',
    weights=original_output_layer.get_weights()  # reuse weights and biases from the original output layer
)(penultimate_output)

continuous_score_model = Model(inputs=model.input, outputs=logits_output)

continuous_score_model.summary()


# ---
# ## Analyse Test Data
# 
# Next, load independent test data (images, labels, age, and timer-positive values) from `.npy` and CSV files.  
# - Timer data is scaled, and age data is processed using `log_scale_age`.  
# - Labels are converted into organ names ("Spleen" or "Thymus"), then encoded into integer and one-hot formats.  
# - The number of samples per class is printed, and separate subsets for spleen and thymus are created, along with their corresponding ages.
# - Visualisation
# - Regression Analysis
# 

# In[4]:


#Import Independent Test Data

base_export_dir = 'Continious_Score_Model_Results'
os.makedirs(base_export_dir, exist_ok=True)

all_images = np.load('test_data/sample_images.npy')
all_labels = np.load('test_data/sample_labels.npy')
age_data = pd.read_csv('test_data/sampledef_age.csv')
timer_pos_data = pd.read_csv('test_data/timer_pos.csv')
timer_pos_data = timer_pos_data['timer_positive'].values
all_ages = age_data['age'].values
all_timers_scaled = timer_pos_data/100
age_scaled = log_scale_age(age_data['age'], 400) 

new_labels = []
for i in range(len(all_labels)):
    organ = "Spleen" if all_labels[i,0] == 1 else "Thymus"

    new_labels.append(organ)

new_labels = np.array(new_labels)

encoder = LabelEncoder()
integer_labels = encoder.fit_transform(new_labels)
one_hot_labels = to_categorical(integer_labels, num_classes=2)
class_indices = {class_name: np.where(integer_labels == i)[0] for i, class_name in enumerate(encoder.classes_)}

age_scaled = age_scaled.values.reshape(-1, 1)
all_timers_scaled = all_timers_scaled.reshape(-1, 1)
spleen_indices = np.where(all_labels[:, 0] == 1)[0]
thymus_indices = np.where(all_labels[:, 0] == 0)[0]
spleen_images = all_images[spleen_indices]
thymus_images = all_images[thymus_indices]

spleen_labels = all_labels[spleen_indices]
thymus_labels = all_labels[thymus_indices]

age_vec = age_data['age']
age_data_spleen = age_vec[spleen_indices]
age_data_thymus = age_vec[thymus_indices]


continuous_scores = continuous_score_model.predict([all_images, age_scaled, all_timers_scaled])

continuous_scores = np.array(continuous_scores[:,0]).flatten() * (-1)
spleen_scores = continuous_scores[spleen_indices]
thymus_scores = continuous_scores[thymus_indices]

spleen_scores = np.array(spleen_scores).flatten()
thymus_scores = np.array(thymus_scores).flatten()
# --- Data Preparation ---
df_spleen = pd.DataFrame({
    'Age': age_data_spleen,
    'Score': spleen_scores,
    'Class': 'Spleen'
})
df_thymus = pd.DataFrame({
    'Age': age_data_thymus,
    'Score': thymus_scores,
    'Class': 'Thymus'
})

df_combined = pd.concat([df_spleen, df_thymus], ignore_index=True)
df_combined['Logged Age'] = np.log2(df_combined['Age'])

# --- Plotting Setup ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
label_font = 22
title_font = 22

# Use all existing ages for vertical grid lines
unique_ages = np.array([1,2, 3,4, 7, 14, 21, 35,70,140 ])#np.unique(df_combined['Age'])
logged_unique_ages = np.log2(unique_ages)

# Panel 0: Original Age vs Score
sns.scatterplot(
    data=df_combined, x='Age', y='Score', hue='Class',
    palette={'Spleen': 'red', 'Thymus': 'blue'},
    alpha=0.4, edgecolor=None, ax=axes[0], s=100
)
axes[0].set_xlabel("Age", fontsize=label_font)
axes[0].set_ylabel("Thymus-Spleen Model Score", fontsize=label_font)
axes[0].legend(title='Class', fontsize=18, title_fontsize=18, loc='upper right')
axes[0].set_xticks(unique_ages) 
axes[0].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
axes[0].tick_params(axis='y', labelsize=16)
axes[0].tick_params(axis='x', labelsize=16)

# Panel 2: Logged Age vs Score (Custom Ticks)
sns.scatterplot(
    data=df_combined, x='Logged Age', y='Score', hue='Class',
    palette={'Spleen': 'red', 'Thymus': 'blue'},
    alpha=0.4, edgecolor=None, ax=axes[1], s=100,
    legend=False
)
axes[1].set_xlabel("Age (log2)", fontsize=label_font)
axes[1].set_ylabel("Thymus-Spleen Model Score", fontsize=label_font)
axes[1].set_xticks(logged_unique_ages)
axes[1].set_xticklabels(unique_ages.astype(int), fontsize = 16)
axes[1].tick_params(axis='y', labelsize=16)  # Set the font size for y-tick labels to 16
axes[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

plt.tight_layout()

os.makedirs(base_export_dir, exist_ok=True)
pdf_path = os.path.join(base_export_dir, 'TestData_class_plot.pdf')
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig)
print(f"Saved the combined plot to {pdf_path}")

csv_path = os.path.join(base_export_dir, 'TestData_class_plot.csv')
df_combined.to_csv(csv_path, index=False)
print(f"Saved the combined DataFrame to {csv_path}")

plt.show()


# In[5]:


# Regression
# Compute continuous scores
continuous_scores = continuous_score_model.predict([all_images, age_scaled, all_timers_scaled])
continuous_scores = np.array(continuous_scores[:, 0]).flatten() * (-1)
spleen_scores = continuous_scores[spleen_indices]
thymus_scores = continuous_scores[thymus_indices]
spleen_scores = np.array(spleen_scores).flatten()
thymus_scores = np.array(thymus_scores).flatten()

# --- Data Preparation ---
df_spleen = pd.DataFrame({
    'Age': age_data_spleen,
    'Score': spleen_scores,
    'Class': 'Spleen'
})
df_thymus = pd.DataFrame({
    'Age': age_data_thymus,
    'Score': thymus_scores,
    'Class': 'Thymus'
})
df_combined = pd.concat([df_spleen, df_thymus], ignore_index=True)
df_combined['Logged Age'] = np.log2(df_combined['Age'])

# --- Quadratic Regression Setup ---
def fit_quadratic_model(X, y):
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear', LinearRegression())
    ])
    model.fit(X, y)
    return model

min_age, max_age = df_combined['Age'].min(), df_combined['Age'].max()
min_logged, max_logged = df_combined['Logged Age'].min(), df_combined['Logged Age'].max()

age_grid = np.linspace(min_age, max_age, 100).reshape(-1, 1)
logged_age_grid = np.linspace(min_logged, max_logged, 100).reshape(-1, 1)

age_grid_df = pd.DataFrame(age_grid, columns=['Age'])
logged_age_grid_df = pd.DataFrame(logged_age_grid, columns=['Logged Age'])

models_age = {}
models_logged = {}
for cl in ['Spleen', 'Thymus']:
    df_cl = df_combined[df_combined['Class'] == cl]
    models_age[cl] = fit_quadratic_model(df_cl[['Age']], df_cl['Score'])
    models_logged[cl] = fit_quadratic_model(df_cl[['Logged Age']], df_cl['Score'])

predictions_age = {cl: models_age[cl].predict(age_grid_df) for cl in ['Spleen', 'Thymus']}
predictions_logged = {cl: models_logged[cl].predict(logged_age_grid_df) for cl in ['Spleen', 'Thymus']}


fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
label_font = 22

unique_ages = np.array([1, 2, 3, 4, 7, 14, 21, 35, 70, 140])
logged_unique_ages = np.log2(unique_ages)

# Panel 0: Raw Age vs Score with Quadratic Regression Lines
sns.scatterplot(
    data=df_combined, x='Age', y='Score', hue='Class',
    palette={'Spleen': 'red', 'Thymus': 'blue'},
    alpha=0.4, edgecolor=None, ax=axes[0], s=100, legend = None
)
for cl, color in zip(['Spleen', 'Thymus'], ['red', 'blue']):
    axes[0].plot(age_grid, predictions_age[cl], color=color, lw=2, label=f'{cl} Quad Fit')
axes[0].set_xlabel("Age", fontsize=label_font)
axes[0].set_ylabel("Thymus-Spleen Model Score", fontsize=label_font)
axes[0].set_xticks(unique_ages)
axes[0].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
axes[0].tick_params(axis='x', labelsize=16)
axes[0].tick_params(axis='y', labelsize=16)

# Panel 1: Logged Age vs Score with Quadratic Regression Lines
sns.scatterplot(
    data=df_combined, x='Logged Age', y='Score', hue='Class',
    palette={'Spleen': 'red', 'Thymus': 'blue'},
    alpha=0.4, edgecolor=None, ax=axes[1], s=100,
    legend=False
)
for cl, color in zip(['Spleen', 'Thymus'], ['red', 'blue']):
    axes[1].plot(logged_age_grid, predictions_logged[cl], color=color, lw=2)
axes[1].set_ylabel("Thymus-Spleen Model Score", fontsize=label_font)
axes[1].set_xlabel("Age (log2)", fontsize=label_font)
axes[1].set_xticks(logged_unique_ages)
axes[1].set_xticklabels(unique_ages.astype(int), fontsize=16)
axes[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
axes[1].tick_params(axis='y', labelsize=16)

plt.tight_layout()


os.makedirs(base_export_dir, exist_ok=True)
pdf_path = os.path.join(base_export_dir, 'TestData_class_plot_with_regression.pdf')
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig)
print(f"Saved the combined plot to {pdf_path}")

csv_path = os.path.join(base_export_dir, 'TestData_class_plot_with_regression.csv')
df_combined.to_csv(csv_path, index=False)
print(f"Saved the combined DataFrame to {csv_path}")

# --- Model Parameters Export ---
parameters = []
for model_type, model_dict, col_name in [('Age', models_age, 'Age'), ('Logged Age', models_logged, 'Logged Age')]:
    for cl, model in model_dict.items():
        poly = model.named_steps['poly']
        linear = model.named_steps['linear']
        feature_names = poly.get_feature_names_out([col_name])
        intercept = linear.intercept_
        parameters.append({
            'Class': cl,
            'Model_Type': model_type,
            'Feature': 'Intercept',
            'Coefficient': intercept
        })
        for fname, coef in zip(feature_names, linear.coef_):
            parameters.append({
                'Class': cl,
                'Model_Type': model_type,
                'Feature': fname,
                'Coefficient': coef
            })


params_df = pd.DataFrame(parameters)
params_csv_path = os.path.join(base_export_dir, 'quadratic_model_parameters.csv')
params_df.to_csv(params_csv_path, index=False)


plt.show()


# In[6]:


# Thymus Data vs Age
X = sm.add_constant(np.column_stack((df_thymus['Age'], df_thymus['Age']**2)))
y = df_thymus['Score']

q_model = sm.OLS(y, X).fit()
print(q_model.summary())


# In[7]:


# Thymus Data vs Log2(Age)
df_thymus['Logged Age'] = np.log2(df_thymus['Age'])
X = sm.add_constant(np.column_stack((df_thymus['Logged Age'], df_thymus['Logged Age']**2)))
y = df_thymus['Score']

q_model = sm.OLS(y, X).fit()

print(q_model.summary())


# In[8]:


# Spleen Data vs Age

X = sm.add_constant(np.column_stack((df_spleen['Age'], df_spleen['Age']**2)))
y = df_spleen['Score']

q_model = sm.OLS(y, X).fit()

print(q_model.summary())


# In[9]:


# Spleen Data vs Log2(Age)
df_spleen['Logged Age'] = np.log2(df_spleen['Age'])
X = sm.add_constant(np.column_stack((df_spleen['Logged Age'], df_spleen['Logged Age']**2)))
y = df_spleen['Score']

q_model = sm.OLS(y, X).fit()

print(q_model.summary())


# In[10]:


continuous_score_model.save('Foxp3DevAge_Logit.keras')


# In[ ]:




