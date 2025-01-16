Project Title: CNN Autoencoder Project (Supervised and Unsupervised)

Purpose:
The project implements both supervised and unsupervised convolutional autoencoders using the CIFAR-10 dataset. It explores the use of autoencoders for dimensionality reduction and image reconstruction, with an additional focus on supervised learning for classification tasks.

Key Components:
1. **DataLoaderUtility Class**:
   - Handles loading and transformation of CIFAR-10 images to PyTorch tensors.
   - Provides functionality for selecting a subset of data based on specific classes and labels.
   
2. **SupervisedAutoencoder Class**:
   - A convolutional autoencoder for image reconstruction with an additional classification layer.
   - The encoder compresses the image into a lower-dimensional representation, while the decoder reconstructs the image.
   
3. **CNN_Autoencoder Class (Unsupervised)**:
   - A similar convolutional autoencoder for image reconstruction without the classification component.
   
4. **SupervisedAutoencoderTrainer Class**:
   - Handles the training of the supervised autoencoder, including both reconstruction loss and classification loss.

5. **CNN_AutoencoderTrainer Class (Unsupervised)**:
   - Trains the unsupervised autoencoder using reconstruction loss.

6. **PlotUtility Class**:
   - Plots the embeddings of images for visualisation purposes, showing how images from different classes are distributed in the learned feature space.

Insights and Analysis:
- **Supervised Autoencoder**: The model learns to reconstruct images while also classifying them into one of three selected classes. The performance of the classifier improves as the model learns to compress and represent image data effectively in the latent space.
- **Unsupervised Autoencoder**: Focuses on image reconstruction without supervision, which is useful for tasks like denoising, feature extraction, or anomaly detection.

Significance:
This project demonstrates how convolutional neural networks can be extended for both unsupervised and supervised autoencoding tasks. The combination of autoencoding for feature extraction and classification allows for efficient learning and practical applications in areas like image compression, anomaly detection, and transfer learning.

Skills Highlighted:
- Deep learning with PyTorch
- Convolutional neural networks (CNN)
- Autoencoder models (Supervised and Unsupervised)
- Image preprocessing and data augmentation
- Training deep learning models and evaluating performance
- Visualisation of embeddings for analysis

How to Run the Project:
1. Ensure that you have installed the necessary dependencies such as PyTorch, Matplotlib, and torchvision.
2. Run the provided Python script. The model will train for 1000 epochs (you can modify the number of epochs as needed).
3. The training process will display loss updates every 100 epochs.
4. After training, embeddings for both the training and test images will be plotted for visualisation.

Expected Input/Output:
- **Input**: CIFAR-10 images are automatically loaded from the internet. The data is preprocessed into PyTorch tensors.
- **Output**: The loss of the autoencoder and classifier (for supervised) is printed during training. After training, a plot of the learned embeddings is displayed.
