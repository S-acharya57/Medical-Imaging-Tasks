# Medical-Imaging-Tasks
This repository has solutions to some medical imaging tasks for better understanding and strengthening the fundamentals.

## 1. Image Processing Task 
This task was to process the 3D scan of a knee, to separate top and bottom bones, after segmenting them successfully. Only thresholding was used at first, with further basic techniques.

## 2. Machine Learning Task
This task was to obtain high metrics for binary classification. The dataset had 300 rows, and around 3000 columns. Thus, feature importances were computed and only top 50 features were obtained, which were then used to train LogisticRegression, SVM model, and ensemble model of them both. The accuracy reached to about 70% in all 3 models. Different techniques like Lasso Regularization, PCA, Bayes Classifier Models were also explored in the process. The metrics are inside the .pdf file inside the directory.

## 3. Deep Learning Task 
This task was to use the segmented regions of 2 bones, and the background. And then to inflate convolution layers of pretrained DenseNet121 model, to make it handle 3D input tensors. And with inference by the regions, the feature maps for the last, third-last, and fifth-last convolution layers were extracted, and average pooled. Converting the feature into fixed N dimensional vector (32 in my case), cosine similarities were computed between each of those 3 regions, for all the selected layers. The metrics obtained are inside the .pdf file inside the directory. 