
from sklearn import svm
#This function will train a linear SVM for every category (i.e. one vs all)
#and then use the learned linear classifiers to predict the category of
#every test image. Every test feature will be evaluated with all SVMs
#and the most confident SVM will "win". Confidence, or distance from the
#margin, is W*X + B where '*' is the inner product or dot product and W and
#B are the learned hyperplane parameters.

def svm_classify(train_image_feats, train_labels, test_image_feats):
    
    
    categories = list(set(train_labels))
    num_categories = len(categories)
    
    #make an SVM classifier
#     clf = svm.LinearSVC()

    #fit on the training data
    #you need to put your own array names here
#     clf.fit(data, labels)

    
    return predicted_categories

# image_feats is an N x d matrix, where d is the dimensionality of the
#  feature representation.
# train_labels is an N x 1 cell array, where each entry is a string
#  indicating the ground truth category for each training image.
# test_image_feats is an M x d matrix, where d is the dimensionality of the
#  feature representation. You can assume M = N unless you've modified the
#  starter code.
# predicted_categories is an M x 1 cell array, where each entry is a string
#  indicating the predicted category for each test image.
