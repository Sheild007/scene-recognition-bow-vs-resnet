import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
import pandas as pd
import os
import glob
import seaborn as sns



def get_image_paths(data_path, categories):
    train_image_paths, train_labels = [], []
    for cat in categories:
        imgs = glob.glob(data_path+'train/'+cat+'/*.*')
        train_image_paths = train_image_paths + imgs
        train_labels = train_labels + [cat]*len(imgs)

    test_image_paths, test_labels = [], []
    for cat in categories:
        imgs = glob.glob(data_path+'test/'+cat+'/*.*')
        if len(imgs) > 0:  
            test_image_paths = test_image_paths + imgs
            test_labels = test_labels + [cat]*len(imgs)

    return np.array(train_image_paths), np.array(test_image_paths), np.array(train_labels), np.array(test_labels)  

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def perf_measure(y_actual, y_hat):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        elif y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        elif y_actual[i]==y_hat[i]==0:
            TN += 1
        elif y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    return [TP, FP, TN, FN]

def display_results(test_labels, categories, predicted_categories, save_path=None, feature_type='', model_type='', vocab_size=None):
    """Display results with accuracy, F1-score, and confusion matrix (matching run_all_tests.py logic)."""
    
    # Calculate accuracy and F1-score
    accuracy = accuracy_score(test_labels, predicted_categories)
    f1 = f1_score(test_labels, predicted_categories, average='macro')
    
    print(f'Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}')
    print(f'Correct predictions: {np.sum(test_labels == predicted_categories)}/{len(test_labels)}\n')
    
    # Create TP/FP/TN/FN table
    df = pd.DataFrame(columns= ['Category']+list(categories))
    cols = ['Category']+['TP', 'FP', 'TN', 'FN']
    df = pd.DataFrame(columns=cols)
    
    for el in categories:
        temp_y_test = (test_labels == el).astype(int)
        temp_preds = (predicted_categories == el).astype(int)
        row = [el] + perf_measure(temp_y_test, temp_preds)
        df = pd.concat([df, pd.DataFrame([row], columns=cols)], ignore_index=True)
    
    print(df, '\n\n')

    # Save confusion matrix if save_path is provided
    if save_path:
        cm_dir = os.path.join(save_path, 'confusion_matrices')
        os.makedirs(cm_dir, exist_ok=True)
        
        # Create filename based on feature and model type
        if vocab_size:
            filename = f'cm_{feature_type}_{model_type}_vocab_{vocab_size}.png'
            title = f'{feature_type.upper()} (vocab={vocab_size}) + {model_type.upper()}'
        else:
            filename = f'cm_{feature_type}_{model_type}.png'
            title = f'{feature_type.upper()} + {model_type.upper()}'
        
        filepath = os.path.join(cm_dir, filename)
        
        # Use seaborn for better visualization like run_all_tests.py
        unique_labels = sorted(np.unique(test_labels))
        save_confusion_matrix_sns(test_labels, predicted_categories, unique_labels, title, filepath)
    else:
        # For displaying, use the old matplotlib method
        test_labels_copy = test_labels.copy()
        predicted_categories_copy = predicted_categories.copy()
        
        for i in range(len(categories)):
            test_labels_copy[test_labels_copy==categories[i]] = i
            predicted_categories_copy[predicted_categories_copy==categories[i]] = i
        test_labels_copy, predicted_categories_copy = test_labels_copy.astype(int), predicted_categories_copy.astype(int)
        
        class_names = np.array(categories)
        plot_confusion_matrix(test_labels_copy, predicted_categories_copy, classes=class_names)
        plt.show()
    
    return

def save_confusion_matrix(y_true, y_pred, labels, title, save_path):
    "
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved confusion matrix to {save_path}")

def plot_tsne(features, labels, title, save_path):
   
    print(f"  Creating t-SNE plot: {title}")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    unique_labels = np.unique(labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_idx[label] for label in labels])
    
    plt.figure(figsize=(14, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                    c=[colors[i]], label=label, s=20, alpha=0.6, edgecolors='black', linewidths=0.2)
    
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     Saved to {save_path}")

