from DataLoader_Resnet import CustomImageDataset
from Resnet_Backbone import Resnet
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pickle
import os
import utils
from BOVW import build_vocabulary, get_bags_of_sifts, save_features_to_pickle


# =========================================================
# Step 0: Setup parameters, paths, and category info
# =========================================================


# 'resnet18'
FEATURE = 'bag of sift'
CLASSIFIER = 'placeholder'  # options: 'nearest neighbor', 'support vector machine'

data_path = './data/'

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

    print('getting feature maps from resnet')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model = Resnet('resnet18').to(device)
    
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            feats = resnet_model(images)  # [batch, 512]
            # store these feature maps and corresponding labels to an array as they will be the input to your classifer
            # these array will also be needed for tsne plot  
    
    #  tsne plot function is given in Resnet_Backbone.py as display_features


elif FEATURE == 'bag of sift':
   
    # Parameters for vocabulary building
    vocab_size = 200  # Options: 100, 200, 500
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
    predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)

elif CLASSIFIER == 'support vector machine':
    predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)

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
utils.display_results(test_labels, categories, predicted_categories) #given in utils.py
