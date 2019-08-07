# Histopathology
Based on Kaggle Challenge: Histopathologic Cancer Detection




<details><summary>Table of Contents</summary><p>

* [Enviroment](#Enviroment)
* [Downloads](#Downloads)

</p></details><p></p>




## Enviroment

## Downloads

## Project Details

### Datasets and Inputs
The dataset was provided in the kaggle challenge [Histopathologic Cancer Detection - Identify metastatic tissue in histopathologic scans of lymph node sections](https://www.kaggle.com/c/histopathologic-cancer-detection/overview). Based on the [PatchCamelyon (PCam) benchmark dataset](https://github.com/basveeling/pcam), the data for the competition was slightly modified. Due to probabilistic sampling, the original dataset contained duplicates that were removed in the Kaggle dataset. The dataset itself contains a very large number of small histopathologic WSIs for classification, taken from lymph node sections. Every file is labeled with an id, and a csv-file provides the information for all image-ids, weather there is at least one cell with tumor tissue in the inner image region or not. If the label is positive, at least one cell in the 32x32px center of the picture is with tumor tissue. The outer region does not influence the label, it was provided to enable fully-convolutional models that do not use zero-padding. For the project the dataset will be used to train, validate and test an algorithm for detecting metastatic tissue in the center of WSIs. Each file is 96x96px large with three color channels each.

### Evaluation Metrics
For evaluation the receiver operating characteristic (ROC) curve will be used. Therefore, the true positive rate (Recall) and the false positive rate (Inverse Recall) are calculated for different thresholds, i.e. splitting points, and the results are allocated in a 2-dimensional diagram. The bigger the area under curve (AUC), the better the classification of the data by the algorithm (with AUC > 0.5, since 0.5 is a random split). Especially in the context of medical treatments it has huge consequences if a disease is correctly identified or not. Hence an optimal classification algorithm should split up sick and heathy persons perfectly. However, since this is not always possible, the performance of a classifier should be measured within the context of identifying as much sick and as much healthy persons as possible. How big the quotas of identified sick, healthy and misclassified persons should be are medical, ethical, and financial decisions. Therefore, the ROC/AUC metrics are very suitable to determine the performance of a binary classifier in this context, plotting the true positive rate against the false positive rate. It illustrates the performance of the classifier at varying thresholds and gives an overall performance parameter, without incorporating any decisions about the optimal threshold for this particular treatment.

### Benchmark
In the [conference paper, which presents the underlying PCam dataset](http://arxiv.org/abs/1806.03962), the best CNN implemented by the authors (P4M-DenseNet) scores 0.963 (AUC), which can be seen as a fair benchmark for this project.

### Data Preparation and Labelling
The original dataset from the kaggle challenge comes as a zip-folder, containing all image data as well as a csv-file with the corresponding labels. In total there are 220.025 single tiff-images, from which a dataframe was created and data was splitted in training/testing (80/20) and then in training/validation (80/20). The data was then stored in labeled folders for further processing. As a result, three folders with labeled subfolders were created, containing 140,816 training, 35,204 validation, and 44,005 testing images.

### Data Distribution and Sample Illustration
Training and testing samples are splited in the ratio 80 to 20 and the remaining 80% training samples were again splited in a training and a validation set in the ratio 80 to 20.

|   | Total | Positive  | Negative  | Ratio  |
|---|---|---|---|---|
| Training | 140,816 | 57,035 | 83,781 | approx. 40/60 |
| Validation | 35,204 | 14,259 | 20,945 | approx. 40/60 |
| Test | 44,005 | 17,823 | 26,182 | approx. 40/60 |

The following two figures illustrate examples from the dataset labeled as non-cancerous in the 30x30 px center and as cancerous in its center. For non-trained ordinary persons, it is hardly possible to identify any differences.

![Non-cancerous](https://github.com/Doc-Ix/histopathology/blob/master/pictures/Example_Images_label_0.png)
*Sample images labeled non-cancerous*


### Data Augmentation and Pipeline

### Model Building

### Results

## License

## Acknowledgments

Komura, D., Ishikawa, S., Machine Learning Methods for Histopathological Image Analysis, Computational and Structural Biotechnology Journal, 16 (2018)

B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation Equivariant CNNs for Digital Pathology", MICCAI (2018), arXiv:1806.03962


https://drive.google.com/open?id=1ptHLjlXdwikzP0kmG7nvswMLJgTpqrsb
