# Histopathology
In this repository you find the capstone project of my Machine Learning Nanodegree at Udacity. Based on the kaggle challenge [Histopathologic Cancer Detection - Identify metastatic tissue in histopathologic scans of lymph node sections](https://www.kaggle.com/c/histopathologic-cancer-detection/overview), the goal of this computer-vision project was an algorithm that is able to identify metastatic cancer in small image patches taken from larger digital pathology scans. Using digital whole slide images (WSI) of lymph nodes sections as input, algorithms were trained to identify tumerous tissue in the center of the WSI provided. The best CNN, based on a Xception network, received a ROC-AUC score of 0.974 on the testing set. Please see section [Project Details](#Project_Details) for further explanation.

Please feel free to use the files provided for further testing/training/processing and experimentation. :smiley:

<details><summary>Table of Contents</summary><p>

* [Content of Repository & Downloads](#Content_Downloads)
* [Enviroment](#Enviroment)
* [Project Details](#Project_Details)
* [License](#License)
* [Acknowledgments](#Acknowledgments)

</p></details><p></p>

<a id="Content_Downloads"></a>
## Content of Repository & Downloads

- **Histo-App-Final.ipynb:** A jupyter notebook file with the main code of the project
- **utils.py:** A python file with supplementary functions
- **environment.yml:** Conda environment file

The following files can be downloaded via google drive:<br>

| File | Description | Link  |
| --- | --- | --- |
|train.zip, train_labels.csv | Original dataset and labels from kaggle challenge | [Link](https://www.kaggle.com/c/histopathologic-cancer-detection/data) |
| weights.best.model_Xception_full.h5 | Weights of best perfoming model, based on Xception | [Link](https://drive.google.com/file/d/16fvWFbsK1SUVJQ9b-NeTdfqrSV1ls4oD/view?usp=sharing) |
| test_data_lables_and_prediction_NASNetmobile_full.pkl | Data frame with the true labels of the testing set as well as the prediction results of NASNetmobile, stored as pickle file. | [Link](https://drive.google.com/file/d/10QSME9fMrpoq767RcBYVKroBWKFgRCZC/view?usp=sharing) |
| test_data_lables_and_prediction_Xception_full.pkl | Data frame with the true labels of the testing set as well as the prediction results of Xception, stored as pickle file. | [Link](https://drive.google.com/file/d/14iIfZtPijTbpQx4cND7IEvmxChyupYzZ/view?usp=sharing) |

<a id="Environment"></a>
## Enviroment
[Create a new Anaconda environment from file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

NAME: tensorflow_p36<br>
`conda env create -f environment.yml`

<a id="Project_Details"></a>
## Project Details 

### Datasets and Inputs
The dataset was provided in the kaggle challenge [Histopathologic Cancer Detection - Identify metastatic tissue in histopathologic scans of lymph node sections](https://www.kaggle.com/c/histopathologic-cancer-detection/overview). Based on the [PatchCamelyon (PCam) benchmark dataset](https://github.com/basveeling/pcam), the data for the competition was slightly modified. Due to probabilistic sampling, the original dataset contained duplicates that were removed in the Kaggle dataset. The dataset itself contains a very large number of small histopathologic WSIs for classification, taken from lymph node sections. Every file is labeled with an id, and a csv-file provides the information for all image-ids, weather there is at least one cell with tumor tissue in the inner image region or not. If the label is positive, at least one cell in the 32x32px center of the picture is with tumor tissue. The outer region does not influence the label, it was provided to enable fully-convolutional models that do not use zero-padding. Each file is 96x96px large with three color channels each.

### Evaluation Metrics
For evaluation the receiver operating characteristic (ROC) curve will be used, plotting the true positive rate against the false positive rate. It illustrates the performance of the classifier at varying thresholds and gives an overall performance parameter, without incorporating any decisions about the optimal threshold for this particular disease resp. treatment.

### Benchmark
In the [conference paper, which presents the underlying PCam dataset](http://arxiv.org/abs/1806.03962), the best CNN implemented by the authors (P4M-DenseNet) scores 0.963 (AUC), which can be seen as a fair benchmark for this project.

### Data Preparation and Labelling
The original dataset from the kaggle challenge comes as a zip-folder, containing all image data as well as a csv-file with the corresponding labels. In total there are 220.025 single tiff-images, from which a dataframe was created and data was splitted in training/testing (80/20) and then in training/validation (80/20). The data was then stored in labeled folders for further processing. As a result, three folders with labeled subfolders were created, containing 140,816 training, 35,204 validation, and 44,005 testing images.

### Data Distribution and Sample Illustration

|   | Total | Positive  | Negative  | Ratio  |
|---|---|---|---|---|
| Training | 140,816 | 57,035 | 83,781 | approx. 40/60 |
| Validation | 35,204 | 14,259 | 20,945 | approx. 40/60 |
| Test | 44,005 | 17,823 | 26,182 | approx. 40/60 |

The following two figures illustrate examples from the dataset labeled as non-cancerous in the 30x30 px center and as cancerous in its center. For non-trained ordinary persons, it is hardly possible to identify any differences.

**Sample images labeled non-cancerous:**
![Non-cancerous](https://github.com/Doc-Ix/histopathology/blob/master/pictures/Example_Images_label_0.png)

**Sample images labeled cancerous:**
![Cancerous](https://github.com/Doc-Ix/histopathology/blob/master/pictures/Example_Images_label_1.png)



### Data Augmentation and Pipeline
Although the dataset is quite large, an additional data augmentation was implemented. Using the Keras module, horizontal as well as vertical flips were implemented as well as rotations and zooms. Furthermore, shifts in height, width, channel as well as shear modulations were realized. To facilitate CNN training, all images were rescaled to values between 0 and 1.
During training and validation, the images should be directly streamed from their corresponding folders, keeping their original size of 96x96 px. With a batch size of 200, data generators for training and validation were initialized. 

### Model Building
Building on pre-trained models in the Keras library and inspired by different blogs about this topic, the following model for training was chosen and implemented, based on a [blog post by Youness Mansar](https://towardsdatascience.com/metastasis-detection-using-cnns-transfer-learning-and-data-augmentation-684761347b59). The core of the network is the NASNetmobile model, since it is credited to be fast and still high performant in image recognition. Three parallel layers follow the core model, a global-max-pooling a global-average-pooling and a flatten layer. After that a dropout layer (rate=0.5) is installed and the final dense layer with a sigmoid activation function builds the end of the model. The resulting CNN has 4,281,333 parameters from which 4,244,595 were trainable in the configuration defined.<br>

**CNN 1, building on NASNetmobile:**

<img src="https://github.com/Doc-Ix/histopathology/blob/master/pictures/model_NASNetmobile_full.png">

As a second network, an Xception model was embedded in a model structure with a global-average-pooling layer and a dropout layer (rate=0.5) following the core model. The final layer is also a dense layer with a sigmoid activation function. With 20,809,001 trainable parameters (out of 20,863,529) the resulting model is much bigger then the NasNetmobile, however much closer to the standard library model.<br>

**CNN 2, building on Xception:**

<img src="https://github.com/Doc-Ix/histopathology/blob/master/pictures/model_Xception.png" width="400">


### Training
The whole models were trained on an AWS EC2 p2.xlarge instance. The training of the models was tracked and the configurations with the currently best validation results were automatically stored to dedicated folders as h5-files.

For detailed training logs, please see the [jupyter notebook file](https://github.com/Doc-Ix/histopathology/blob/master/Histo-App.ipynb).

### Testing
In order to make predictions with the trained models, data frames with the paths to the single testing images and their corresponding labels were created. The predictions of the models were then added to a new column of the data frame and the data frames were stored as pickle files, after deleting unnecessary columns, in order to reduce file size. The data frames for the two models can be found in the [Downloads](#Content_Downloads) section. After that the data frames were optimized for further processing and an additional column with a binary prediction value was added, using a threshold of 0.5.

### Results
| Model | Sensitivity | Specitivity | False Positive Rate | ROC-AUC |
| --- | --- | --- | --- | --- |
| CNN 1, building on NASNetmobile | 81.06 % | 90.22 % | 9.78 % | 0.937 |
| CNN 2, building on Xception | 83.23 % | 97.78 % | 2.22 % | 0.974 |

With an ROC-AUC score of 0.974 the CNN 2 model was able to reach the goal of the project, to build a binary image classifier for histopathologic cancer detecting, reaching an ROC-AUC value above 0.95.

**Plot of ROC-AUC for Xception model:**

![ROC-AUC_Xception](https://github.com/Doc-Ix/histopathology/blob/master/pictures/ROC_Xception_full.png)

<a id="License"></a>
## License
This repository is under the [MIT License](https://choosealicense.com/licenses/mit/).

<a id="Acknowledgments"></a>
## Acknowledgments
* The dataset of this project was provided by kaggle, building on the dataset of Bas Veeling, with additional input from Babak Ehteshami Bejnordi, Geert Litjens, and Jeroen van der Laak.

* Thanks to kaggle for hosting this challenge
