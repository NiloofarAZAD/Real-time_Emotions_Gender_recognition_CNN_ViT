# Real-time Facial Expression and Gender Recognition Using CNNs and Vision Transformers
Implementing an image classification model using a combination of ResNet50 and Vision Transformers for a real-time facial expression and gender recognition system.

⚡ Facial Expressions Dateset: https://www.kaggle.com/datasets/msambare/fer2013

After 100 epochs, the ViT model achieves around 78.21% accuracy on the test data. However, this is not a competitive result on a rather small dataset compared to CNNs on the same data, which achieved 91% accuracy. CNNs achieve excellent results even with training based on data volumes that are not as large as those required by Vision Transformers.

The ViT model in the original paper is trained on the JFT-300M dataset and then fine-tuned on the target dataset. To receive high accuracy, it's recommended to train a ViT on a large, high-resolution dataset.
