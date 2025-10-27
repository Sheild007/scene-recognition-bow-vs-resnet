# Roll Number: BSCS22008
# Name: Muhammad Usman Muneer
# Assignment Number: 3


import os
import argparse
import pickle
import glob
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader


import utils
from DataLoader_Resnet import CustomImageDataset
from Resnet_Backbone import Resnet
from BOVW import (
    build_vocabulary, 
    get_bags_of_sifts, 
    save_features_to_pickle,
    plot_histogram, 
    visualize_sift_keypoints
)
from SVM import svm_classify
from KNN import nearest_neighbor_classify




parser = argparse.ArgumentParser(description='Scene Recognition using BoW or ResNet features with KNN or SVM classifier')
parser.add_argument('--input', type=str, default='./data/', help='Input folder containing train and test data')
parser.add_argument('--output', type=str, default='./results_mine/', help='Output folder for results')
parser.add_argument('--feature', type=str, choices=['bow', 'resnet'], default='bow', 
                    help='Feature type: bag of words (bow) or resnet')
parser.add_argument('--model', type=str, choices=['knn', 'svm'], default='svm', 
                    help='Classification model: knn or svm')
parser.add_argument('--vocab_size', type=int, default=200, 
                    help='Vocabulary size for bag of words (default: 200, options: 100, 200, 500)')

args = parser.parse_args()


FEATURE = 'bag of sift' if args.feature == 'bow' else 'resnet'
CLASSIFIER = 'nearest neighbor' if args.model == 'knn' else 'support vector machine'

data_path = args.input
output_path = args.output
vocab_size = args.vocab_size


os.makedirs(output_path, exist_ok=True)

categories = np.array([
    'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
    'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
    'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest'
])

#get image paths is given in utils.py
print('Getting paths and labels for all train and test data\n')
train_image_paths, test_image_paths, train_labels, test_labels = utils.get_image_paths(data_path, categories)

#   train_image_paths  1500x1   cell      
#   test_image_paths   705x1    cell           
#   train_labels       1500x1   cell         
#   test_labels        705x1    cell          



# Map string label -> int
categories = sorted(list(set(train_labels)))   # ensure fixed order
class_to_idx = {cat: idx for idx, cat in enumerate(categories)}


# =========================================================
# Step 1: Represent each image with appropriate feature
# =========================================================



print("Using", FEATURE, "representation for images\n")

if FEATURE == 'resnet':

        # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Create datasets
    train_dataset = CustomImageDataset(train_image_paths, train_labels, class_to_idx, transform=transform)
    test_dataset  = CustomImageDataset(test_image_paths, test_labels, class_to_idx, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #you can change batch size accordingly
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("Num classes:", len(categories))
    print("Train size:", len(train_dataset))
    print("Test size:", len(test_dataset))

    print('Getting feature maps from resnet')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model = Resnet('resnet18').to(device)
    resnet_model.eval()
    
    # Extract features from training images
    train_image_feats = []
    print('Extracting features from training images...')
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            feats = resnet_model(images)  # [batch, 512]
            train_image_feats.append(feats.cpu().numpy())
    
    train_image_feats = np.vstack(train_image_feats)
    print(f'Train features shape: {train_image_feats.shape}')
    
    # Extract features from test images
    test_image_feats = []
    print('Extracting features from test images...')
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            feats = resnet_model(images)  # [batch, 512]
            test_image_feats.append(feats.cpu().numpy())
    
    test_image_feats = np.vstack(test_image_feats)
    print(f'Test features shape: {test_image_feats.shape}')
    
    # Generate t-SNE visualization for ResNet features
    print('\nGenerating t-SNE visualization...')
    tsne_path = os.path.join(output_path, 'tsne_resnet.png')
    resnet_model.display_features(train_image_feats, train_labels, 
                                  title="t-SNE visualization of ResNet features (Training)",
                                  save_path=tsne_path)


elif FEATURE == 'bag of sift':
   
    # Parameters for vocabulary building
    max_features_per_image = 300  # Options: 200, 300, 400
    
    save_path = f'vocab_size_{vocab_size}.pkl'
    
    if(not os.path.exists(save_path)):
        print('No existing visual word vocabulary found. Computing one from training images\n')
        print(f'Parameters: vocab_size={vocab_size}, max_features_per_image={max_features_per_image}')
        vocab = build_vocabulary(train_image_paths, vocab_size, 
                                max_features_per_image=max_features_per_image,
                                save_path=save_path) #given in BOVW.py
    else:
        print(f'Loading existing vocabulary from {save_path}')
        with open(save_path, 'rb') as f:
            vocab = pickle.load(f)
     
    # Code get_bags_of_sifts function
    print('Getting bag of sift features for training images...')
    train_image_feats = get_bags_of_sifts(train_image_paths, vocab_size=vocab_size) #given in BOVW.py
    print('Getting bag of sift features for test images...')
    test_image_feats  = get_bags_of_sifts(test_image_paths, vocab_size=vocab_size)
    
    # Save features to pickle file
    feat_save_path = f'bow_features_vocab_{vocab_size}.pkl'
    save_features_to_pickle({
        'train_features': train_image_feats,
        'test_features': test_image_feats,
        'train_labels': train_labels,
        'test_labels': test_labels
    }, feat_save_path)
    
    # Generate visualizations: histograms and SIFT keypoints
    print('\nGenerating visualizations...')
    
    # Plot histograms for sample images (one per category)
    hist_dir = os.path.join(output_path, 'histograms')
    os.makedirs(hist_dir, exist_ok=True)
    
    sample_categories = ['Bedroom', 'Coast', 'Forest', 'Kitchen', 'Store']
    for category in sample_categories:
        if category in train_labels:
            # Find first image of this category
            indices = np.where(train_labels == category)[0]
            if len(indices) > 0:
                sample_idx = indices[0]
                sample_path = train_image_paths[sample_idx]
                
                # Extract features for this image
                sample_feats = get_bags_of_sifts([sample_path], vocab_size=vocab_size)
                
                # Plot histogram
                hist_path = os.path.join(hist_dir, f'hist_{category}.png')
                plot_histogram(sample_feats[0], vocab_size, hist_path, 
                              title=f'BoW Histogram - {category} (vocab_size={vocab_size})')
    
    # Visualize SIFT keypoints for sample images
    sift_dir = os.path.join(output_path, 'keypoints')
    os.makedirs(sift_dir, exist_ok=True)
    
    for category in sample_categories:
        if category in train_labels:
            indices = np.where(train_labels == category)[0]
            if len(indices) > 0:
                sample_idx = indices[0]
                sample_path = train_image_paths[sample_idx]
                
                # Visualize SIFT keypoints
                sift_path = os.path.join(sift_dir, f'sift_{category}.png')
                visualize_sift_keypoints(sample_path, sift_path, max_keypoints=500)
    
    # Generate t-SNE visualization for BoW features
    print('\nGenerating t-SNE visualization for BoW features...')
    tsne_path = os.path.join(output_path, f'tsne_bovw_vocab_{vocab_size}.png')
    
    # Use ResNet's display_features method to avoid code duplication
    # Create a temporary ResNet instance to use its display_features method
    resnet_model_temp = Resnet('resnet18')
    
    # Sample a subset for t-SNE (it's computationally expensive)
    n_samples = min(1000, len(train_image_feats))
    indices = np.random.choice(len(train_image_feats), n_samples, replace=False)
    sampled_feats = train_image_feats[indices]
    sampled_labels = train_labels[indices]
    
    resnet_model_temp.display_features(
        sampled_feats, 
        sampled_labels, 
        title=f'Bag of Visual Words - t-SNE Visualization (vocab_size={vocab_size})',
        save_path=tsne_path
    )
   

elif FEATURE == 'placeholder':
    train_image_feats = []
    test_image_feats = []

else:
    print("Unknown feature type")




# =========================================================
# Step 2: Train Classifier and Predict
# =========================================================

# Classify each test image by training and using the appropriate classifier
#  to classify test features should return an N x 1 cell array,
# where N is the number of test cases and each entry is a string indicating
# the predicted category for each test image. Each entry in
# 'predicted_categories' must be one of the 15 strings in 'categories',
# 'train_labels', and 'test_labels'. See the starter code for each function
# for more details.


print('Using', CLASSIFIER, 'classifier to predict test set categories\n')

if CLASSIFIER == 'nearest neighbor':
    # Test with different k values
    k_values = [1, 3, 5, 7, 9, 11, 15]
    all_results = {}
    
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Testing with k={k}")
        print(f"{'='*60}\n")
        predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, k=k)
        
        # Calculate accuracy
        accuracy = np.sum(predicted_categories == test_labels) / len(test_labels)
        all_results[k] = {'predictions': predicted_categories, 'accuracy': accuracy}
        print(f"\nAccuracy for k={k}: {accuracy:.4f}")
        print(f"{'='*60}\n")
    
    # Use k=3 as default
    predicted_categories = all_results[3]['predictions']

elif CLASSIFIER == 'support vector machine':
    # Determine model save path based on feature type
    if FEATURE == 'bag of sift':
        model_path = os.path.join(output_path, f'svm_model_bovw_vocab_{vocab_size}.pkl')
    else:
        model_path = os.path.join(output_path, 'svm_model_resnet.pkl')
    
    predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats, 
                                        save_model=True, model_path=model_path)

elif CLASSIFIER == 'placeholder':
    # Random guessing for debugging
    predicted_categories = np.random.choice(categories, size=len(test_labels))

else:
    print("Unknown classifier type")



# =========================================================
# Step 3: Display results
# =========================================================

## Step 3: Build a confusion matrix and score the recognition system
# You do not need to code anything in this section. 

# If we wanted to evaluate our recognition method properly we would train
# and test on many random splits of the data. You are not required to do so
# for this assignment.

# This function will plot confusion matrix and accuracy of your model
feature_str = 'bovw' if FEATURE == 'bag of sift' else 'resnet'
model_str = 'knn' if CLASSIFIER == 'nearest neighbor' else 'svm'
vocab_size_param = vocab_size if FEATURE == 'bag of sift' else None

utils.display_results(test_labels, categories, predicted_categories, 
                      save_path=output_path, 
                      feature_type=feature_str, 
                      model_type=model_str,
                      vocab_size=vocab_size_param)


# =========================================================
# Step 8: Testing Functions
# =========================================================

def extract_bovw_features_single(image_path, vocab_path):
    """Extract BoVW features from a single image."""
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    vocab_size = vocab.shape[0]
    features = get_bags_of_sifts([image_path], vocab_size=None)
    return features[0]


def extract_resnet_features_single(image_path, model, device='cpu'):
    """Extract ResNet features from a single image - helper for testing functions."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        features = model(img_tensor)
    
    return features.cpu().numpy().flatten()


def testing_onOneImage(image_path, classifier_type, model_path=None, bow_vocab_path='vocab_size_200.pkl'):
    """Test a single image with either KNN or SVM classifier."""
    print("\n" + "="*70)
    print(f"TESTING SINGLE IMAGE")
    print("="*70)
    print(f"Image path: {image_path}")
    print(f"Classifier: {classifier_type}")
    print(f"{'='*70}\n")
    
    # Load BoVW features
    try:
        features_file = f'bow_features_vocab_200.pkl'
        with open(features_file, 'rb') as f:
            train_data = pickle.load(f)
            train_image_feats_bovw = train_data['train_features']
            train_labels_data = train_data['train_labels']
    except FileNotFoundError:
        print("Error: BoVW features not found. Please run main.py first.")
        return None
    
    print("Extracting features from image...")
    image_feats_bovw = extract_bovw_features_single(image_path, bow_vocab_path)
    
    print(f"\nClassifying with {classifier_type}...")
    
    if classifier_type == 'KNN':
        k = 3
        predicted_label_bovw = nearest_neighbor_classify(
            train_image_feats_bovw.reshape(len(train_image_feats_bovw), -1),
            train_labels_data,
            image_feats_bovw.reshape(1, -1),
            k=k
        )
        print(f"\nPredicted label (BoVW+KNN): {predicted_label_bovw[0]}")
        return predicted_label_bovw[0]
        
    elif classifier_type == 'SVM':
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                svm_model = model_data['model']
            
            predicted_label_bovw = svm_model.predict(image_feats_bovw.reshape(1, -1))
            print(f"\nPredicted label (BoVW+SVM): {predicted_label_bovw[0]}")
            return predicted_label_bovw[0]
        else:
            print(f"Error: Model file not found at {model_path}")
            return None
    else:
        print("Error: classifier_type must be 'KNN' or 'SVM'")
        return None


def testing_AllImages(image_folder_path, classifier_type, model_path=None, bow_vocab_path='vocab_size_200.pkl'):
    """Test all images in a folder and compute overall accuracy and confusion matrix."""
    print("\n" + "="*70)
    print(f"TESTING ALL IMAGES IN FOLDER")
    print("="*70)
    print(f"Image folder: {image_folder_path}")
    print(f"Classifier: {classifier_type}")
    print(f"{'='*70}\n")
    
    # Get all image paths
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(image_folder_path, '**', ext), recursive=True))
    
    if len(image_paths) == 0:
        print(f"Error: No images found in {image_folder_path}")
        return None, None
    
    print(f"Found {len(image_paths)} images to test\n")
    
    # Load BoVW features
    try:
        features_file = f'bow_features_vocab_200.pkl'
        with open(features_file, 'rb') as f:
            train_data = pickle.load(f)
            train_image_feats_bovw = train_data['train_features']
            train_labels_data = train_data['train_labels']
    except FileNotFoundError:
        print("Error: BoVW features not found. Please run main.py first.")
        return None, None
    
    # Extract features for all test images
    print("Extracting features from all images...")
    test_features_bovw = []
    true_labels = []
    
    for idx, img_path in enumerate(image_paths):
        if (idx + 1) % 10 == 0:
            print(f"Processing image {idx + 1}/{len(image_paths)}")
        
        # Extract BoVW features
        feat = extract_bovw_features_single(img_path, bow_vocab_path)
        test_features_bovw.append(feat)
        
        # Get ground truth label from folder name
        folder_name = os.path.basename(os.path.dirname(img_path))
        true_labels.append(folder_name)
    
    test_features_bovw = np.array(test_features_bovw)
    
    # Classify all images
    print(f"\nClassifying all images with {classifier_type}...")
    predicted_labels = []
    
    if classifier_type == 'KNN':
        k = 3
        for feat in test_features_bovw:
            pred = nearest_neighbor_classify(
                train_image_feats_bovw.reshape(len(train_image_feats_bovw), -1),
                train_labels_data,
                feat.reshape(1, -1),
                k=k
            )
            predicted_labels.append(pred[0])
    
    elif classifier_type == 'SVM':
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                svm_model = model_data['model']
            
            predicted_labels = svm_model.predict(test_features_bovw)
        else:
            print(f"Error: Model file not found at {model_path}")
            return None, None
    
    # Calculate accuracy
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"  Correct: {np.sum(true_labels == predicted_labels)}/{len(true_labels)}")
    
    # Create and save confusion matrix
    unique_labels_list = sorted(np.unique(np.hstack([true_labels, predicted_labels])))
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_labels_list)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {classifier_type}', fontsize=14, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(unique_labels_list))
    plt.xticks(tick_marks, unique_labels_list, rotation=45, ha='right')
    plt.yticks(tick_marks, unique_labels_list)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    save_path = f'results/confusion_matrix_{classifier_type}.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()
    
    print("\n" + "="*70)
    print("Classification Report:")
    print("="*70)
    print(classification_report(true_labels, predicted_labels, labels=unique_labels_list))
    print("="*70 + "\n")
    
    return accuracy, cm
