
#This function will sample SIFT descriptors from the training images,
#cluster them with kmeans, and then return the cluster centers.

def build_vocabulary(image_paths, vocab_size):
    
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




def get_bags_of_sifts(image_paths):
    
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
    
    
    
    
    
    return image_feats

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



