# Script Classification Project

The evolution of writing systems has led to a diverse array of scripts, many structurally linked to their ancient predecessors, and understanding these relationships is crucial for studying the evolution of language and culture. This project aims to classify handwritten characters from three major historical parent scripts---Brahmi, Classical Chinese, and Phoenician---using a Convolutional Neural Network (CNN). Initially, the classifier will be trained and tested on labeled data from these scripts:

- A mixture of ancient Brahmi characters and Devanagari: class Brahmi
- A mixture of ancient Phoenician, ancient Greek, and Arabic characters: class Phoenician
- Classical Chinese

The model's accuracy on this task will provide the foundation for using it to classify any given character from any script to these three families based on the structural features of each parent class that is learned while training. This is based on the rational that, though different script underwent different transformations over its evolution, they still retain certain structural components from their parent scripts. By first training the models on a combination of parent-daughter scripts(Brahmi-Devanagari, Phoenician-Greek,Arabic), the model learns to focus on features similar across the evolution that allows them to classify both the parent and daughter under the same class. Once this training is complete, the model can extrapolate these learned features into even unseen daughter scripts and classify them to their parent script accurately(refer to examples in the dataset directory for samples).

## **Input and Output**

The input to the model is a 64x64 grayscale image of a handwritten character in .png/jpeg/bmp format. The output is a probability distribution across the three parent scripts, representing the likelihood of the scripts origin over the three script families.


## **Implementation philosophy**

The model is trained on a subset of scripts from each parent class and a daughter class, aiming covering all the structural diversity in the script family. Now, when a character from a script unseen in the training but belonging to the same family is used, the model tries to classify it among the three parent classes, based on structural similaritis like troke orientation, stroke thickness, stroke density, geometric layout, symmetry, number of strokes, number of intersections, curvature patterns, contour patterns, stroke endpoints etc. 

- **Brahmi**: Devanagari, Bengali, Tamil, Kannada, Telugu, Malayalam etc.
- **Phoenician**: Greek, Aramaic, Hebrew, Arabic, Syriac, Latin, Cyrillic, etc.
- **Classical Chinese (Hanzi)**: Japanese Kanji, Korean Hanja, Vietnamese Nôm, Seal, Regular, Cursive

**NOTE: Make sure that the images used for evaluation are handwritten and not stylised. Since the model was trained on handwritten characters, using stylised versions might give unreliable outputs**
**NOTE: Make sure that the script characters are sufficiently far from the image border**

## **Data Sources**

**Brahmi**: Kaggle dataset: [Brahmi dataset | Kaggle](https://www.kaggle.com/datasets/gautamneha/brahmi-dataset)

**Devanagari:** Kaggle dataset: https://www.kaggle.com/datasets/medahmedkrichen/devanagari-handwritten-character-datase

**Classical Chinese**: CASIA-AHCDB - Chinese Ancient Handwritten Characters Database: [Home People Projects Publications Seminar Activities Data&Codes CASIA-AHCDB: Chinese Ancient Handwritten Characters Database](https://nlpr.ia.ac.cn/pal/CASIA-AHCDB.html)

**Ancient Greek:** Kaggle dataset: https://www.kaggle.com/datasets/vrushalipatel/handwritten-greek-characters-from-gcdb

**Arabic:** Kaggle dataset: https://www.kaggle.com/datasets/mloey1/ahcd1

**Phoenician**: HPCDB, Mendeley Data: [Handwritten Phoenician Character DataBase (HPCDB) - Mendeley Data](https://data.mendeley.com/datasets/x5prh2hyj8/1)

## **Model Architecture**

The model is based on a custom CNN architecture that combines standard convolutional layers with Inception-style modules to efficiently extract multi-scale features from 64x64 directional feature maps with four input channels.

The first convolutional layer applies 32 filters with a 9x9 kernel (stride 2) and ReLU activation, capturing broad stroke patterns while reducing spatial dimensions. The second convolutional block increases the depth to 64 filters with a 3x3 kernel, further refining local feature extraction such as edges and intersections.

After these initial layers, the network introduces two sequential Inception modules. Each Inception module performs parallel convolutions with different kernel sizes (1x1, 3x3, 5x5) and max-pooling operations, allowing the model to simultaneously learn fine and coarse features. The first Inception module receives 64 input channels and outputs 80 channels, while the second expands to 80 input channels and outputs 80 channels again, enabling deep multi-scale feature fusion.

Following feature extraction, the architecture employs both Global Average Pooling (GAP) and Global Max Pooling (GMP) layers to condense spatial information, which are concatenated to form a robust feature vector. This pooled output is flattened and passed through a fully connected layer with 256 neurons, using ReLU activation and 30% dropout regularization to mitigate overfitting.

The final classification layer outputs three neurons, corresponding to the three parent script classes ("Brahmi", "Classical Chinese", and "Phoenician"), with softmax activation implicitly applied during loss computation.

To handle potential class imbalances in the dataset, the training pipeline includes weighted sampling based on class frequencies, ensuring fair representation of all classes during optimization. This architecture is designed to balance multi-scale feature extraction, computational efficiency, and robustness, making it suitable for both CPU and GPU environments.

## **Imlementation Overview**
This section briefly describes the key modules and functions, including their inputs and expected outputs.

### config.py
- **Global Variables:**  
  Contains hyperparameters and settings such as `batchsize`, `epochs`, `resize_x`, `resize_y`, `input_channels`, `learning_rate`, etc.

### dataset.py
- **CustomDataset(train_dir, test_dir, val_split=0.2)**  
  - **Input:**  
    - `train_dir`: Path to the training image folder  
    - `test_dir`: Path to the test image folder  
    - `val_split`: (Optional) Validation split ratio (default: 0.2)  
  - **Output:**  
    - Creates training, validation (with class balancing), and test datasets.
- **get_loaders()**  
  - **Output:**  
    - Returns a tuple `(train_loader, val_loader, test_loader)` as PyTorch DataLoaders.
- **get_dataloader(train_dir, test_dir)**  
  - **Behavior:**  
    - Convenience function that instantiates `CustomDataset` and returns its DataLoaders.

### model.py
- **InceptionModule(in_channels)**  
  - **Input:**  
    - Number of input channels.  
  - **Output:**  
    - Applies parallel convolutions (with kernels of varying sizes) and pooling, concatenating the outputs.
- **ScriptCNN(num_classes=3)**  
  - **Input:**  
    - Expects images of size 64×64 with 4 channels.  
  - **Output:**  
    - Produces logits for three classes via a combination of standard convolutional layers, Inception modules, global pooling, and a dense layer.

### train.py
**NOTE: the training and test directory have the following structure: root--->train/test--->classes(Brahmi/Classical Chinese/Phoenician)--->character_folder--->image files**
- **set_eval_loaders(val_dl, test_dl)**  
  - **Input:**  
    - `val_dl`: Validation DataLoader  
    - `test_dl`: Test DataLoader  
  - **Behavior:**  
    - Sets global placeholders for evaluation during training.
- **evaluate(model, dataloader, loss_fn)**  
  - **Output:**  
    - Returns accuracy, average loss, and lists of predictions and true labels.
- **train_model(model, num_epochs, train_loader, loss_fn, optimizer)**  
  - **Behavior:**  
    - Executes the training loop over multiple epochs (including forward pass, loss computation, backward propagation, and optimizer updates).  
    - Logs metrics and saves the best model based on validation loss improvements.
  - **Output:**  
    - Returns the trained model.

### predict.py
- **classify_images(list_of_image_paths)**  
  - **Input:**  
    - A list of image file paths (from the `data/` directory).  
  - **Output:**  
    - For each image, returns a dictionary containing:  
      - `image_path`: The image file path  
      - `predicted_label`: The predicted class label  
      - `confidence`: Prediction confidence for the label  
      - `probabilities`: The full probability distribution over the three classes

### interface.py
- **Purpose:**  
  Standardizes function and class names for automated grading by mapping:
  - `TheModel` → `ScriptCNN`
  - `the_trainer` → `train_model`
  - `the_predictor` → `classify_images`
  - `TheDataset` → `CustomDataset`
  - `the_dataloader` → `get_dataloader`
  - Also imports hyperparameters (`the_batch_size`, `total_epochs`) from `config.py`.

---
