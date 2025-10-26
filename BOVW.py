import cv2
import numpy as np
from sklearn.cluster import KMeans
import pickle
import os
import matplotlib.pyplot as plt

#This function will sample SIFT descriptors from the training images,
#cluster them with kmeans, and then return the cluster centers.

def build_vocabulary(image_paths, vocab_size, max_features_per_image=None, save_path=None):
    """
    Sample SIFT descriptors from training images, cluster them with k-means.
    
    Args:
        image_paths: List of image paths
        vocab_size: Number of clusters (visual words)
        max_features_per_image: Maximum features to sample per image
        save_path: Path to save the vocabulary
    
    Returns:
        vocab: vocab_size x 128 matrix of cluster centroids
    """
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
    
    all_descriptors = np.vstack(all_descriptors)
    print(f"Total descriptors collected: {len(all_descriptors)}")
    
    # Cluster with k-means
    print(f"Clustering {len(all_descriptors)} descriptors into {vocab_size} visual words...")
    kmeans = KMeans(n_clusters=vocab_size, random_state=42, n_init=15)
    kmeans.fit(all_descriptors)
    
    vocab = kmeans.cluster_centers_
    print(f"Vocabulary shape: {vocab.shape}")

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Vocabulary saved to {save_path}")
    
    return vocab

# The inputs are 'image_paths', a N x 1 cell array of image paths, and
# 'vocab_size' the size of the vocabulary.

# The output 'vocab' should be vocab_size x 128. Each row is a cluster
# centroid / visual word.


# Load images from the training set. To save computation time, you don't
# necessarily need to sample from all images, although it would be better
# to do so. You can randomly sample the descriptors from each image to save
# memory and speed up the clustering. 

# For each loaded image, get some SIFT features. You don't have to get as
# many SIFT features as you will in get_bags_of_sift, because you're only
# trying to get a representative sample here.

# Once you have tens of thousands of SIFT features from many training
# images, cluster them with kmeans. The resulting centroids are now your
# visual word vocabulary.




def get_bags_of_sifts(image_paths, vocab_size=None):
    """
    Represent each image as a histogram of visual word occurrences.
    
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
    
    # Randomly sample between 400-500 descriptors from each image
    np.random.seed(42)  # For reproducibility
    
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            image_feats.append(np.zeros(actual_vocab_size))
        else:
            keypoints, descriptors = sift.detectAndCompute(img, None)
            
            if descriptors is None or len(descriptors) == 0:
                image_feats.append(np.zeros(actual_vocab_size))
            else:
                # Randomly sample 400-500 descriptors from each image
                descriptors_per_image = np.random.randint(400, 501)  # 400-500 range
                if len(descriptors) > descriptors_per_image:
                    indices = np.random.choice(len(descriptors), descriptors_per_image, replace=False)
                    descriptors = descriptors[indices]
                
                # Compute L2 distances to all visual words
                distances = np.linalg.norm(descriptors[:, np.newaxis, :] - vocab[np.newaxis, :, :], axis=2)
                
                # Assign each descriptor to nearest visual word
                nearest_clusters = np.argmin(distances, axis=1)
                
                # Build histogram (count of descriptors assigned to each visual word)
                histogram = np.bincount(nearest_clusters, minlength=actual_vocab_size)
                
                # Normalize the histogram
                if histogram.sum() > 0:
                    histogram = histogram / histogram.sum()
                
                image_feats.append(histogram)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_paths)} images")
    
    image_feats = np.array(image_feats)
    print(f"Created features with shape: {image_feats.shape}")
    
    return image_feats
# Use SIFT from Open-CV library refer to the code below for help and update perameters
# to install open-cv use following commands
# pip install opencv-python
# pip install opencv-contrib-python

#     sift = cv2.xfeatures2d.SIFT_create(30) #specify how many maximum descriptors you want in the output 
#     im = Image.open(img_path)
#     im.thumbnail(self.out_size, Image.ANTIALIAS) 
#     img = np.array(im)

#     kp, des = sift.detectAndCompute(img, None)
    
    # vocab = pickle.load('vocab.pkl')
    # vocab_size = 
    
    
    
    
    
   

# image_paths is an N x 1 cell array of strings where each string is an
# image path on the file system.

# This function assumes that 'vocab.pkl' exists and contains an N x 128
# matrix 'vocab' where each row is a kmeans centroid or visual word. This
# matrix is saved to disk rather than passed in a parameter to avoid
# recomputing the vocabulary in every run.

# image_feats is an N x d matrix, where d is the dimensionality of the
# feature representation. In this case, d will equal the number of clusters
# or equivalently the number of entries in each image's histogram
# ('vocab_size') below.

# You will want to construct SIFT features here in the same way you
# did in build_vocabulary function (except for possibly changing the sampling
# rate) and then assign each local feature to its nearest cluster center
# and build a histogram indicating how many times each cluster was used.
# Don't forget to normalize the histogram, or else a larger image with more
# SIFT features will look very different from a smaller version of the same
# image.

#  SIFT_features is a 128 x N matrix of SIFT features
#   note: there are smoothing parameters you can manipulate for sift function


def plot_histogram(histogram, vocab_size, save_path=None, title=None):
    """
    Visualize the histogram of visual word occurrences.
    
    Args:
        histogram: Array of histogram values (1 x vocab_size)
        vocab_size: Size of vocabulary
        save_path: Path to save the plot
        title: Title for the plot
    """
    plt.figure(figsize=(12, 6))
    plt.bar(range(vocab_size), histogram)
    plt.xlabel('Visual Word Index')
    plt.ylabel('Frequency')
    if title:
        plt.title(title)
    else:
        plt.title(f'Bag of Visual Words Histogram (vocab_size={vocab_size})')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Histogram saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_sift_keypoints(image_path, save_path=None, max_keypoints=50):
    """
    Visualize SIFT keypoints on an image.
    
    Args:
        image_path: Path to the image
        save_path: Path to save the visualization
        max_keypoints: Maximum number of keypoints to display
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert to color for visualization
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(img, None)
    
    # Draw keypoints on the image
    img_with_keypoints = cv2.drawKeypoints(
        img_color, 
        keypoints, 
        None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # Create figure and display
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title(f'SIFT Keypoints Visualization\nNumber of keypoints: {len(keypoints)}')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"SIFT keypoints visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_features_to_pickle(features, file_path):
    """
    Save image features to a pickle file.
    
    Args:
        features: Array of features to save
        file_path: Path to save the pickle file
    """
    with open(file_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"Features saved to {file_path}")


def load_features_from_pickle(file_path):
    """
    Load image features from a pickle file.
    
    Args:
        file_path: Path to the pickle file
    
    Returns:
        features: Array of features
    """
    with open(file_path, 'rb') as f:
        features = pickle.load(f)
    print(f"Features loaded from {file_path}")
    return features



