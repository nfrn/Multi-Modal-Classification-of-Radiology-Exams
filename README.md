# Multi-Modal Approach Based on Deep Convolutional Networks for the Classification of Chest Radiology Examinations
This work was developed in the context of a MSc thesis at Instituto Superior Técnico, University of Lisbon.
The code is not meant to be used for commercial purposes.

The source code in this project leverages the keras.io deep learning libray for implementing a deep neural network that combines word embeddings,
recurrent units, and neural attention, together with state-of-the-art CNN networks, for the task of classify Chest Radiology Examinations.

The complete multi-modal model with pre-trained weights (i.e. image processing path pre-trained with CheXpert and MIMIC-CXR, and text modeling path pre-trained with MIMIC-III dataset) and fine-tuned with the open-i training split, achieved a very high performance in terms of the different metrics. Pre-training, in particular, contributed significantly to the overall performance of the complete model.

For further information about the method, the reader can refer to the following publication who reported early results:
@inproceedings{nunes2019multi,
  title={A Multi-modal Deep Learning Method for Classifying Chest Radiology Exams},
  author={Nunes, Nelson and Martins, Bruno and da Silva, Nuno Andr{\'e} and Leite, Francisca and Silva, M{\'a}rio J},
  booktitle={EPIA Conference on Artificial Intelligence},
  pages={323--335},
  year={2019},
  organization={Springer}
}

# Instructions:
  *The Dataset folder provides the scripts necessary to preprocess the MIMIC-CXR, MIMIC-III, and Open-i datasets described in our experiments.
  
  *The Text_Model, Image_Model folders provide scripts to create the models, train, and generate predictions leveraging the correspondent datasets.
  
  *The Merged_Model folder provide scripts to create the complete model with and without pre-trained weights, train, generate predictions, and provide the visualization mechanisms(Grad-CAM and Word Weights) leveraging the Open-i dataset.
  
  *The Evaluation folder uses the predicted numpy array files generated from the previous folders and computes the multi-label evaluation metrics.
  


