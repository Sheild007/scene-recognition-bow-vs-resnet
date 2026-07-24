# Scene Recognition — Bag of Visual Words vs. ResNet Features

A comparison between classical BoVW (SIFT + k-means vocabulary + SVM/KNN) and deep ResNet50 features for scene classification. The goal was to understand where the classical pipeline holds up and where it falls apart compared to learned representations.

## Methods

### Bag of Visual Words
- SIFT keypoints extracted from training images
- K-means clustering to build a visual vocabulary (tested at 100, 200, 500 words)
- Each image represented as a histogram of nearest-vocabulary-word counts
- Classifiers: SVM (RBF kernel) and KNN

### ResNet50 Features
- Pretrained ResNet50 with the classification head removed
- 2048-dim feature vector extracted per image
- Same SVM and KNN classifiers applied on top

Both feature types are evaluated with the same train/test split so the comparison is fair.

## Running experiments

```bash
# BoVW with SVM (vocab size 200)
python main.py --input ./data/ --output ./results --feature bow --model svm --vocab_size 200

# BoVW with KNN
python main.py --input ./data/ --output ./results --feature bow --model knn --vocab_size 200

# ResNet features with SVM
python main.py --input ./data/ --output ./results --feature resnet --model svm

# ResNet features with KNN
python main.py --input ./data/ --output ./results --feature resnet --model knn
```

## Testing on custom images

```bash
# Single image
python main.py --test_single path/to/image.jpg --test_type SVM \
    --test_model_path ./results/svm_model_bow.pkl \
    --vocab_path vocab_size_200.pkl
```

## Vocabulary pre-generation

Generating vocabulary is the slow step. Pre-compute it once:

```bash
python generate_vocabularies.py
```

Saved as `vocab_size_100.pkl`, `vocab_size_200.pkl`, `vocab_size_500.pkl`.

## What I'd improve

BoVW loses spatial information completely — a spatial pyramid would help with scenes where layout matters (e.g. kitchen vs. office). The ResNet features also aren't fine-tuned, so there's a gap between what it can do and what we're measuring.
