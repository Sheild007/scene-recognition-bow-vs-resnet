import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 
# 3-channel RGB images of shape (3xHxW)
# where H and W are expected to be atleast 224
# The images have to be loaded in to a range of [0,1]
# and then normalized using  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

# Your dataloader_resnet.py already handles all the preprocessing mentioned above , but feel free to ge through the code 
# for your knowledge

class Resnet(nn.Module):
    def __init__(self, resnet_variant='resnet18'):
        super(Resnet, self).__init__()
        
        # Load pretrained resnet backbone
        self.model = torch.hub.load('pytorch/vision:v0.10.0', resnet_variant, pretrained=True)
        
        # Remove the final fully connected layer (classifier)
        # This keeps convolutional feature extractor only (output: 512-dim feature for resnet18
        self.features = nn.Sequential(*list(self.model.children())[:-1])  
       
    def forward(self, x):
        # Forward pass through feature extractor
        x = self.features(x)   # output shape: (batch_size, 512, 1, 1) for resnet18
        x = torch.flatten(x, 1)  # flatten to (batch_size, 512)
        return x
    
    def display_features(self,features_list,labels_list):
        
        self.eval() #setting model to eval mode

        features_list = []
        labels_list = []

        # Stack all batches
        features = np.vstack(features_list)
        labels = np.hstack(labels_list)

        print("Feature shape before t-SNE:", features.shape)

        # Apply t-SNE (reduce 512-dim → 2D for plotting)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)

        # Plot
        plt.figure(figsize=(10,8))
        scatter = plt.scatter(features_2d[:,0], features_2d[:,1], c=labels, cmap='tab10', s=15, alpha=0.7)
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.title("t-SNE visualization of ResNet features")
        plt.show()
