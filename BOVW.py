import cv2
import numpy as np
from sklearn.cluster import KMeans
import pickle

#This function will sample SIFT descriptors from the training images,
#cluster them with kmeans, and then return the cluster centers.

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
    
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    
        if img is None:
            image_feats.append(np.zeros(actual_vocab_size))
        else:
            keypoints, descriptors = sift.detectAndCompute(img, None)

            if descriptors is None or len(descriptors) == 0:
                image_feats.append(np.zeros(actual_vocab_size))
            else:
            
                distances = np.linalg.norm(descriptors[:, np.newaxis, :] - vocab[np.newaxis, :, :], axis=2)
                nearest_clusters = np.argmin(distances, axis=1)
                histogram = np.bincount(nearest_clusters, minlength=actual_vocab_size)

                if histogram.sum() > 0:
                    histogram = histogram / histogram.sum()

                image_feats.append(histogram)11  

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




