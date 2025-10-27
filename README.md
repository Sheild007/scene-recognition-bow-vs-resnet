# Scene Recognition using BoW and ResNet

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/JuGqlEvf)



## Running the Project

### Training and Evaluation

```bash
# BoVW features with SVM
python main.py --input ./data/ --output ./results --feature bow --model svm --vocab_size 200

# BoVW features with KNN
python main.py --input ./data/ --output ./results --feature bow --model knn --vocab_size 200

# ResNet features with SVM
python main.py --input ./data/ --output ./results --feature resnet --model svm

# ResNet features with KNN
python main.py --input ./data/ --output ./results --feature resnet --model knn
```

### Testing Custom Images

```bash
# Single image with SVM
python main.py --test_single path/to/image.jpg --test_type SVM \
    --test_model_path ./results/svm_model_bow.pkl \
    --vocab_path vocab_size_200.pkl

# Single image with KNN
python main.py --test_single path/to/image.jpg --test_type KNN \
    --vocab_path vocab_size_200.pkl

# Test folder with SVM
python main.py --test_folder ./data/test/ --test_type SVM \
    --test_model_path ./results/svm_model_bow.pkl \
    --vocab_path vocab_size_200.pkl
```
