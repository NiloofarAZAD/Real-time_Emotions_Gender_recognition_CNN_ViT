# Real-time Facial Expression and Gender Recognition Using CNNs and Vision Transformers
A comparison of CNNs and Vision Transformers in image classification for a real-time facial expression and gender recognition system

⚡ Facial Expressions Dateset: https://www.kaggle.com/datasets/msambare/fer2013

After 100 epochs, the ViT model achieves around 78.21% accuracy on the test data. However, this is not a competitive result on a rather small dataset compared to CNNs on the same data, which achieved 91% accuracy. To receive high accuracy, it's recommended to train a ViT on a large, high-resolution dataset.

The results reported in the original paper are achieved by pre-training the ViT model using the JFT-300M dataset and then fine-tuning it on the target dataset. 
