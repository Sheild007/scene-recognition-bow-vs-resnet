# Roll Number: BSCS22008
# Name: Muhammad Usman Muneer
# Assignment Number: 3

import numpy as np
import os
import pickle
import argparse  
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import utils
import BOVW 
from SVM import svm_classify
from KNN import nearest_neighbor_classify
import torch
from torchvision import transforms
from Resnet_Backbone import Resnet
from DataLoader_Resnet import CustomImageDataset
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from utils import save_confusion_matrix, plot_tsne
from PIL import Image
import cv2
from collections import Counter
import glob
from testing import testing_onOneImage, testing_AllImages, extract_features_from_single_image


def extract_resnet_features(data_path, categories, device='cpu'):
  
    train_image_paths, test_image_paths, train_labels, test_labels = utils.get_image_paths(data_path, categories)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Map labels
    categories_list = sorted(list(set(train_labels)))
    class_to_idx = {cat: idx for idx, cat in enumerate(categories_list)}
    
    # Create datasets
    train_dataset = CustomImageDataset(train_image_paths, train_labels, class_to_idx, transform=transform)
    test_dataset = CustomImageDataset(test_image_paths, test_labels, class_to_idx, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load model
    model = Resnet('resnet18').to(device)
    model.eval()
    
    # Extract features
    train_features = []
    test_features = []
    
    print("  Extracting ResNet features from training images...")
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            feats = model(images)
            train_features.append(feats.cpu().numpy())
    train_features = np.vstack(train_features)
    print(f"   Extracted {len(train_features)} training features")
    
    print("  Extracting ResNet features from test images...")
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            feats = model(images)
            test_features.append(feats.cpu().numpy())
    test_features = np.vstack(test_features)
    print(f"   Extracted {len(test_features)} test features")
    
    return train_features, test_features, train_labels, test_labels


def extract_bow_features(data_path, categories, vocab_size):

    vocab_file = f'vocab_size_{vocab_size}.pkl'
    features_file = f'bow_features_vocab_{vocab_size}.pkl'
    
    if os.path.exists(features_file):
        print(f"Loading cached BoW features from {features_file}...")
        try:
            with open(features_file, 'rb') as f:
                data = pickle.load(f)
            print("Cached features loaded.")
            return data['train_features'], data['test_features'], data['train_labels'], data['test_labels']
        except Exception as e:
            print(f"ERROR loading cached features: {e}. Rebuilding...")

    print("Loading image paths...")
    train_image_paths, test_image_paths, train_labels, test_labels = \
        utils.get_image_paths(data_path, categories)
    print(f"Found {len(train_image_paths)} train and {len(test_image_paths)} test images.")

    if not os.path.exists(vocab_file):
        print(f"Building vocabulary (size {vocab_size})...")
        try:
            BOVW.build_vocabulary(train_image_paths, vocab_size, 
                                  max_features_per_image=500, save_path=vocab_file)
            print(f"Vocabulary built and saved to {vocab_file}.")
        except AttributeError:
            print("ERROR: `BOVW.build_vocabulary` not found. Check BOVW.py.")
            return None, None, None, None
        except Exception as e:
            print(f"ERROR building vocabulary: {e}")
            return None, None, None, None
    else:
        print(f"Using existing vocabulary: {vocab_file}")

    print("Extracting train histograms...")
    try:
        train_features = BOVW.get_bags_of_sifts(train_image_paths, vocab_size=vocab_size)
        print(f"Extracted {len(train_features)} train features.")
    except AttributeError:
        print("ERROR: `BOVW.get_bags_of_sifts` not found. Check BOVW.py.")
        return None, None, None, None
    except Exception as e:
        print(f"ERROR building train histograms: {e}")
        return None, None, None, None

    print("Extracting test histograms...")
    try:
        test_features = BOVW.get_bags_of_sifts(test_image_paths, vocab_size=vocab_size)
        print(f"Extracted {len(test_features)} test features.")
    except Exception as e:
        print(f"ERROR building test histograms: {e}")
        return None, None, None, None
    
    print(f"Caching features to {features_file}...")
    try:
        with open(features_file, 'wb') as f:
            pickle.dump({
                'train_features': train_features,
                'test_features': test_features,
                'train_labels': train_labels,
                'test_labels': test_labels
            }, f)
        print(f"Features cached.")
    except Exception as e:
        print(f"Warning: Could not cache features: {e}")
        
    return train_features, test_features, train_labels, test_labels


# Testing functions (extract_features_from_single_image, testing_onOneImage, testing_AllImages) 
# are now imported from testing.py


def main(args):
   
    print("\n" + "="*80)
    print("SCENE RECOGNITION PIPELINE")
    print(f"  Feature:     {args.feature}")
    print(f"  Model:       {args.model}")
    print(f"  Input Path:  {args.input}")
    print(f"  Output Path: {args.output}")
    if args.feature == 'bow':
        print(f"  Vocab Size:  {args.vocab_size}")
    print("="*80 + "\n")


    data_path = args.input
    output_path = args.output
    cm_dir = os.path.join(output_path, 'confusion_matrices')
    
  
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)
    
 
    categories = np.array([
        'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
        'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
        'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest'
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    train_features, test_features, train_labels, test_labels = None, None, None, None
    feature_name_str = "" 
    
    print("\n" + "="*80)
    print("STEP 1: Feature Extraction")
    print("="*80)
    
    if args.feature == 'resnet':
        feature_name_str = "ResNet"
        print(f"Extracting {feature_name_str} features...")
        train_features, test_features, train_labels, test_labels = \
            extract_resnet_features(data_path, categories, device)
        
        print("\n" + "="*80)
        print("STEP 1a: t-SNE Visualization (ResNet)")
        print("="*80)
        sample_size = 500 
        indices = np.random.choice(len(train_features), min(sample_size, len(train_features)), replace=False)
        tsne_save_path = os.path.join(output_path, f'tsne_{args.feature}.png')
        plot_tsne(train_features[indices], train_labels[indices],
                 'ResNet Features t-SNE', tsne_save_path)

    elif args.feature == 'bow':
        feature_name_str = f"BoW_v{args.vocab_size}"
        print(f"Extracting Bag-of-Words (BoW) features (vocab size: {args.vocab_size})...")
        train_features, test_features, train_labels, test_labels = \
            extract_bow_features(data_path, categories, args.vocab_size)

    if train_features is None:
         print("\n Feature extraction failed. Check errors above. Exiting.")
         return

    unique_labels = sorted(np.unique(train_labels))

    print("\n" + "="*80)
    print(f"STEP 2: Classification using {args.model.upper()}")
    print("="*80)
    
    results = {}
    model_name_str = ""
    
    if args.model == 'knn':
        model_name_str = "KNN"
        print(f"\n  Running {feature_name_str} + {model_name_str}...")
        
        # Test with different k values (as in original script)
        knn_results_by_k = {}
        best_acc = -1
        best_k = 1
        
        for k in [1, 3, 5]:
            print(f"    Testing with k={k}...")
            pred = nearest_neighbor_classify(train_features, train_labels, test_features, k=k)
            
            acc = accuracy_score(test_labels, pred)
            f1 = f1_score(test_labels, pred, average='macro')
            
            knn_results_by_k[k] = {
                'predictions': pred,
                'accuracy': acc,
                'f1_score': f1
            }
            print(f"      Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_k = k
        
        print(f"    Best k={best_k} with Accuracy: {best_acc:.4f}")
        
        # Store best result
        method_key = f"{feature_name_str}+{model_name_str}(k={best_k})"
        results[method_key] = knn_results_by_k[best_k]
        
        # Save confusion matrix for best k
        cm_filename = f'cm_{args.feature}_{args.model}_k{best_k}.png'
        cm_save_path = os.path.join(cm_dir, cm_filename)
        save_confusion_matrix(test_labels, knn_results_by_k[best_k]['predictions'], unique_labels,
                             f'{feature_name_str} + {model_name_str} (k={best_k})', cm_save_path)

    elif args.model == 'svm':
        model_name_str = "SVM"
        print(f"\n  Running {feature_name_str} + {model_name_str}...")
        
        model_filename = f'svm_model_{args.feature}.pkl'
        model_save_path = os.path.join(output_path, model_filename)
        
        pred_svm = svm_classify(train_features, train_labels, test_features, 
                                save_model=True, model_path=model_save_path)
        
        acc = accuracy_score(test_labels, pred_svm)
        f1 = f1_score(test_labels, pred_svm, average='macro')
        
        method_key = f"{feature_name_str}+{model_name_str}"
        results[method_key] = {
            'predictions': pred_svm,
            'accuracy': acc,
            'f1_score': f1
        }
        
        print(f"  Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
        
     
        cm_filename = f'cm_{args.feature}_{args.model}.png'
        cm_save_path = os.path.join(cm_dir, cm_filename)
        save_confusion_matrix(test_labels, pred_svm, unique_labels,
                             f'{feature_name_str} + {model_name_str}', cm_save_path)

   
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'Method':<30} {'Accuracy':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    for method, res in results.items():
        print(f"{method:<30} {res['accuracy']:<12.4f} {res['f1_score']:<12.4f}")
    
  
    summary_filename = f'summary_{args.feature}_{args.model}.pkl'
    summary_save_path = os.path.join(output_path, summary_filename)
    
    print(f"\nSaving summary to {summary_save_path}...")
    with open(summary_save_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"   Saved summary.")

    print("\n" + "="*80)
    print(f" Pipeline for {args.feature} + {args.model} completed!")
    print(f" Results saved to '{output_path}' directory")
    print("="*80 + "\n")
    
    return results

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser(description='Scene Recognition using BoW or ResNet features with KNN or SVM classifier')
    parser.add_argument('--input', type=str, default='./data/', 
                        help='Input folder containing train and test data')
    parser.add_argument('--output', type=str, default='./results_mine/', 
                        help='Output folder for results')
    parser.add_argument('--feature', type=str, choices=['bow', 'resnet'], default='bow', 
                        help='Feature type: bag of words (bow) or resnet')
    parser.add_argument('--model', type=str, choices=['knn', 'svm'], default='svm', 
                        help='Classification model: knn or svm')
    parser.add_argument('--vocab_size', type=int, default=200, 
                        help='Vocabulary size for bag of words (default: 200, options: 100, 200, 500)')
    
   
    parser.add_argument('--test_single', type=str, default=None,
                        help='Test a single image. Provide path to image file.')
    parser.add_argument('--test_folder', type=str, default=None,
                        help='Test all images in a folder. Provide path to folder.')
    parser.add_argument('--test_type', type=str, choices=['KNN', 'SVM'], default='SVM',
                        help='Classifier type for testing: KNN or SVM')
    parser.add_argument('--test_model_path', type=str, default=None,
                        help='Path to saved model file (required for SVM, not needed for KNN)')
    parser.add_argument('--vocab_path', type=str, default='vocab_size_200.pkl',
                        help='Path to BoW vocabulary file (default: vocab_size_200.pkl)')

    args = parser.parse_args()

    
    if args.test_single:
     
        testing_onOneImage(
            image_path=args.test_single,
            type=args.test_type,
            path_to_model=args.test_model_path,
            path_to_BOW_VOCAB=args.vocab_path
        )
    elif args.test_folder:
       
        testing_AllImages(
            image_folder_path=args.test_folder,
            type=args.test_type,
            path_to_model=args.test_model_path,
            path_to_BOW_VOCAB=args.vocab_path
        )
    else:
     
        main(args)