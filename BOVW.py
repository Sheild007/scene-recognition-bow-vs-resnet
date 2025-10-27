import cv2
import numpy as np
from sklearn.cluster import KMeans
import pickle
import os
import matplotlib.pyplot as plt


def build_vocabulary(image_paths, vocab_size, max_features_per_image=None, save_path=None):

    sift = cv2.SIFT_create()
    all_descriptors = []
    
    print(f"Extracting SIFT features from {len(image_paths)} images...")
   
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            keypoints, descriptors = sift.detectAndCompute(img, None)
            
            if descriptors is not None:
            
                if max_features_per_image and len(descriptors) > max_features_per_image:
                    indices = np.random.choice(len(descriptors), max_features_per_image, replace=False)
                    descriptors = descriptors[indices]
                
                all_descriptors.append(descriptors)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_paths)} images")
    
    if not all_descriptors:
        print("Error: No descriptors were extracted. Check image paths and content.")
        return None

    all_descriptors = np.vstack(all_descriptors)
    print(f"Total descriptors collected: {len(all_descriptors)}")
    
    print(f"Clustering {len(all_descriptors)} descriptors into {vocab_size} visual words...")
    kmeans = KMeans(n_clusters=vocab_size, random_state=42, n_init=10) 
    kmeans.fit(all_descriptors)
    
    vocab = kmeans.cluster_centers_
    print(f"Vocabulary shape: {vocab.shape}")
    if save_path:
        
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Vocabulary saved to {save_path}")
    
    return vocab

def get_bags_of_sifts(image_paths, vocab_size=None):
    
    if vocab_size:
        vocab_file = f'vocab_size_{vocab_size}.pkl'
    else:
        vocab_file = 'vocab.pkl'
    
    if not os.path.exists(vocab_file):
        print(f"Error: Vocabulary file not found at {vocab_file}")
        return None

    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    
    actual_vocab_size = vocab.shape[0]
    print(f"Loaded vocabulary with {actual_vocab_size} visual words from {vocab_file}")

    sift = cv2.SIFT_create()
    image_feats = []
    
  
    np.random.seed(42)  
    
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not read {img_path}. Appending zero vector.")
            image_feats.append(np.zeros(actual_vocab_size))
        else:
            keypoints, descriptors = sift.detectAndCompute(img, None)
            
            if descriptors is None or len(descriptors) == 0:
             
                image_feats.append(np.zeros(actual_vocab_size))
            else:
           
                descriptors_per_image = np.random.randint(400, 501)  
                if len(descriptors) > descriptors_per_image:
                    indices = np.random.choice(len(descriptors), descriptors_per_image, replace=False)
                    descriptors = descriptors[indices]
               
                distances = np.linalg.norm(descriptors[:, np.newaxis, :] - vocab[np.newaxis, :, :], axis=2)
    
                nearest_clusters = np.argmin(distances, axis=1)
               
              
                histogram = np.bincount(nearest_clusters, minlength=actual_vocab_size)
                
            
                if histogram.sum() > 0:
                    histogram = histogram / histogram.sum()
                
                image_feats.append(histogram)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_paths)} images")
    
    image_feats = np.array(image_feats)
    print(f"Created features with shape: {image_feats.shape}")
    
    return image_feats

def plot_histogram(histogram, vocab_size, save_path=None, title=None):

    plt.figure(figsize=(12, 6))
    plt.bar(range(vocab_size), histogram)
    plt.xlabel('Visual Word Index')
    plt.ylabel('Normalized Frequency')
    if title:
        plt.title(title)
    else:
        plt.title(f'Bag of Visual Words Histogram (vocab_size={vocab_size})')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Histogram saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_sift_keypoints(image_path, save_path=None, max_keypoints=500):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
   
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
  
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
  
    keypoints, descriptors = sift.detectAndCompute(img, None)
    
    img_with_keypoints = cv2.drawKeypoints(
        img_color, 
        keypoints, 
        None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title(f'SIFT Keypoints Visualization\nNumber of keypoints: {len(keypoints)}')
    plt.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"SIFT keypoints visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def save_features_to_pickle(features, file_path):
  
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"Features saved to {file_path}")

def load_features_from_pickle(file_path):

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
        
    with open(file_path, 'rb') as f:
        features = pickle.load(f)
    print(f"Features loaded from {file_path}")
    return features