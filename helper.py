import os
import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def visualize_perform(history):
    """visulize performance

    visualize performance of model training
    How to Use: put the history of training in history parameter
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Visualize the Train Data
    ax1.plot(history.history["loss"], label="Training Loss")
    ax1.plot(history.history["val_loss"], label="Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy and Loss")
    ax1.legend()

    # Visualize the Test Data
    ax2.plot(history.history["accuracy"], label="Training Accuracy")
    ax2.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy and Loss")
    ax2.legend()

    plt.tight_layout() # Adjust
    plt.show()


def visualize_matrix(actual, prediction):
    """Visualize Matrix

    visualize matrix for seing accuracy from the model
    How to use: put the actual data and prediction from the model
    
    its like -> actual==y_true -> prediction==y_pred
    """
    
    cm = confusion_matrix(actual, np.argmax(prediction, axis=1))

    kategori_labels = [f'{i}' for i in range(len(cm))]

    # Membuat visualisasi matriks konfusi dengan Seaborn
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.1, cbar=False, square=True,
                xticklabels=kategori_labels, yticklabels=kategori_labels, cbar_kws={"orientation": "horizontal"})
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Konfusion Matrik")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

def save_model_ml(model, name):
    """Save Model ML
    
    saving model from ml model
    How to use: put the name of model and name for file saving
    """
    
    with open(f'../model/{name}.pkl', 'wb') as file:
        pickle.dump(model, file)

        
def load_model_ml(filepath):
    """Load Model ML
    
    load model from ml model
    How to use: put the file path from model saving
    """
    
    with open(f'{filepath}', 'rb') as file:
        model = pickle.load(file)
    
    return model
        
        
        