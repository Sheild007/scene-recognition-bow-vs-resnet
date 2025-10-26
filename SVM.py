
from sklearn import svm
import numpy as np
import pickle
import os

#This function will train a linear SVM for every category (i.e. one vs all)
#and then use the learned linear classifiers to predict the category of
#every test image. Every test feature will be evaluated with all SVMs
#and the most confident SVM will "win". Confidence, or distance from the
#margin, is W*X + B where '*' is the inner product or dot product and W and
#B are the learned hyperplane parameters.

def svm_classify(train_image_feats, train_labels, test_image_feats, save_model=True, model_path='svm_model.pkl'):
    """
    Classify test images using linear SVM.
    
    Args:
        train_image_feats: N x d matrix of training features
        train_labels: N x 1 array of training labels
        test_image_feats: M x d matrix of test features
        save_model: Whether to save the trained model
        model_path: Path to save the model
    
    Returns:
        predicted_categories: M x 1 array of predicted labels
    """
    categories = list(set(train_labels))
    num_categories = len(categories)
    print(f"Training SVM with {num_categories} categories")
    
    # Make an SVM classifier with one-vs-rest strategy
    clf = svm.LinearSVC(dual=False, random_state=42)
    
    # Fit on the training data
    print("Training SVM classifier...")
    clf.fit(train_image_feats, train_labels)
    print("Training completed!")
    
    # Predict labels for test features
    print("Predicting test image labels...")
    predicted_categories = clf.predict(test_image_feats)
    
    # Save the model if requested
    if save_model:
        model_data = {
            'model': clf,
            'categories': categories,
            'num_categories': num_categories
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")
    
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
