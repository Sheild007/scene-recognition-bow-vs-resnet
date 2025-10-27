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
    
    def display_features(self, features, labels, title="t-SNE visualization", save_path=None):
        """
        Display features using t-SNE visualization.
        
        Args:
            features: numpy array of features (N x d)
            labels: numpy array of labels (N,)
            title: Title for the plot
            save_path: Optional path to save the plot
        """
        self.eval() # setting model to eval mode

        print("Feature shape before t-SNE:", features.shape)
        print(f"Number of samples: {len(features)}")

        # Apply t-SNE (reduce to 2D for plotting)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        features_2d = tsne.fit_transform(features)

        # Convert labels to numeric if they are strings
        unique_labels = np.unique(labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        numeric_labels = np.array([label_to_idx[label] for label in labels])

        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_2d[:,0], features_2d[:,1], c=numeric_labels, 
                             cmap='tab20', s=15, alpha=0.6, edgecolors='black', linewidths=0.1)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)))
        cbar.set_ticklabels(unique_labels)
        
        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"t-SNE plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
