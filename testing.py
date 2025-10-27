import numpy as np
import os
import pickle
import cv2
from collections import Counter
import glob
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def extract_features_from_single_image(image_path, feature_type='bow', vocab_path=None, resnet_model=None, device='cpu'):
 
    if feature_type == 'bow':
        if vocab_path is None:
            print("Error: vocab_path is required for BoW features")
            return None
        
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        
        vocab_size = vocab.shape[0]
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not read {image_path}")
            return np.zeros(vocab_size)
        
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(img, None)
        
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(vocab_size)
        
        descriptors_per_image = np.random.randint(400, 501)
        if len(descriptors) > descriptors_per_image:
            indices = np.random.choice(len(descriptors), descriptors_per_image, replace=False)
            descriptors = descriptors[indices]
        
        distances = np.linalg.norm(descriptors[:, np.newaxis, :] - vocab[np.newaxis, :, :], axis=2)
        nearest_clusters = np.argmin(distances, axis=1)
        
        histogram = np.bincount(nearest_clusters, minlength=vocab_size)
        
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()
        
        return histogram
    
    elif feature_type == 'resnet':
        from PIL import Image
        import torch
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        resnet_model.eval()
        with torch.no_grad():
            features = resnet_model(img_tensor)
        
        return features.cpu().numpy().flatten()
    
    else:
        print(f"Error: Unknown feature_type '{feature_type}'")
        return None


def testing_onOneImage(image_path, type, path_to_model, path_to_BOW_VOCAB):
    
    print("\n" + "="*80)
    print("TESTING SINGLE IMAGE")
    print("="*80)
    print(f"Image: {image_path}")
    print(f"Type: {type}")
    print(f"Model path: {path_to_model}")
    print(f"BoW vocab path: {path_to_BOW_VOCAB}")
    print("="*80)
    
    if not os.path.exists(image_path):
        print(f"\nError: Image not found at {image_path}")
        return None
    
    if not os.path.exists(path_to_BOW_VOCAB):
        print(f"\nError: BoW vocabulary not found at {path_to_BOW_VOCAB}")
        return None
    
    data_path = './data/'
    categories = np.array([
        'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
        'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
        'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest'
    ])
    
    try:
        features_file = 'bow_features_vocab_200.pkl'
        with open(features_file, 'rb') as f:
            train_data = pickle.load(f)
            train_features = train_data['train_features']
            train_labels = train_data['train_labels']
    except FileNotFoundError:
        print(f"\nError: Training features not found. Please run main.py first to generate features.")
        return None
    
    print("\nExtracting BoW features from image...")
    test_features = extract_features_from_single_image(image_path, 'bow', path_to_BOW_VOCAB)
    
    if test_features is None:
        print("Error: Failed to extract features")
        return None
    
    print(f"\nClassifying with {type}...")
    
    if type.upper() == 'KNN':
        k = 3
        distances = np.linalg.norm(train_features - test_features, axis=1)
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = train_labels[k_nearest_indices]
        
        label_counter = Counter(k_nearest_labels)
        predicted_label = label_counter.most_common(1)[0][0]
        
        print(f"\nPredicted Label: {predicted_label}")
        print(f"  Using k={k} nearest neighbors")
        
        return predicted_label
    
    elif type.upper() == 'SVM':
        if path_to_model is None or not os.path.exists(path_to_model):
            print(f"\nError: Model file not found at {path_to_model}")
            return None
        
        with open(path_to_model, 'rb') as f:
            model_data = pickle.load(f)
            svm_model = model_data['model']
        
        predicted_label = svm_model.predict(test_features.reshape(1, -1))[0]
        
        print(f"\nPredicted Label: {predicted_label}")
        
        return predicted_label
    
    else:
        print(f"\nError: type must be 'KNN' or 'SVM', got '{type}'")
        return None


def testing_AllImages(image_folder_path, type, path_to_model, path_to_BOW_VOCAB):
    
    print("\n" + "="*80)
    print("TESTING ALL IMAGES IN FOLDER")
    print("="*80)
    print(f"Folder: {image_folder_path}")
    print(f"Type: {type}")
    print(f"Model path: {path_to_model}")
    print(f"BoW vocab path: {path_to_BOW_VOCAB}")
    print("="*80)
    
    if not os.path.exists(image_folder_path):
        print(f"\nError: Folder not found at {image_folder_path}")
        return None, None
    
    if not os.path.exists(path_to_BOW_VOCAB):
        print(f"\nError: BoW vocabulary not found at {path_to_BOW_VOCAB}")
        return None, None
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(image_folder_path, '**', ext), recursive=True))
    
    if len(image_paths) == 0:
        print(f"\nError: No images found in {image_folder_path}")
        return None, None
    
    print(f"\nFound {len(image_paths)} images to test")
    
    data_path = './data/'
    categories = np.array([
        'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
        'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
        'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest'
    ])
    
    try:
        features_file = 'bow_features_vocab_200.pkl'
        with open(features_file, 'rb') as f:
            train_data = pickle.load(f)
            train_features = train_data['train_features']
            train_labels = train_data['train_labels']
    except FileNotFoundError:
        print(f"\nError: Training features not found. Please run main.py first to generate features.")
        return None, None
    
    print("\nExtracting features from all images...")
    test_features_list = []
    true_labels = []
    
    for idx, img_path in enumerate(image_paths):
        if (idx + 1) % 10 == 0:
            print(f"  Processing {idx + 1}/{len(image_paths)} images")
        
        feat = extract_features_from_single_image(img_path, 'bow', path_to_BOW_VOCAB)
        if feat is not None:
            test_features_list.append(feat)
            
            folder_name = os.path.basename(os.path.dirname(img_path))
            true_labels.append(folder_name)
    
    test_features = np.array(test_features_list)
    true_labels = np.array(true_labels)
    
    print(f"\nExtracted features from {len(test_features)} images")
    
    print(f"\nClassifying all images with {type}...")
    predicted_labels = []
    
    if type.upper() == 'KNN':
        k = 3
        for feat in test_features:
            distances = np.linalg.norm(train_features - feat, axis=1)
            k_nearest_indices = np.argsort(distances)[:k]
            k_nearest_labels = train_labels[k_nearest_indices]
            
            label_counter = Counter(k_nearest_labels)
            predicted_label = label_counter.most_common(1)[0][0]
            predicted_labels.append(predicted_label)
    
    elif type.upper() == 'SVM':
        if path_to_model is None or not os.path.exists(path_to_model):
            print(f"\nError: Model file not found at {path_to_model}")
            return None, None
        
        with open(path_to_model, 'rb') as f:
            model_data = pickle.load(f)
            svm_model = model_data['model']
        
        predicted_labels = svm_model.predict(test_features)
    
    else:
        print(f"\nError: type must be 'KNN' or 'SVM', got '{type}'")
        return None, None
    
    predicted_labels = np.array(predicted_labels)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    
    print(f"\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Correct: {np.sum(true_labels == predicted_labels)}/{len(true_labels)}")
    print("="*80)
    
    unique_labels = sorted(np.unique(np.hstack([true_labels, predicted_labels])))
    save_path = f'results_mine/cm_testing_{type.lower()}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title(f'Confusion Matrix - Testing All Images with {type}', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nConfusion matrix saved to {save_path}")
    
    return accuracy, cm