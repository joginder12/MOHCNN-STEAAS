# Set-theoretic Explainable AI-based Attribution Score for Selecting miRNAs in Pan-cancer Data (MOHCNN-STEAAS)
Identification of miRNAs associated with cancer is a complex computational
task. A multi-objective framework for optimizing hyperparameters of a 1D CNN,
called MOHCNN, and a set-theoretic explainable AI-based attribution score
(STEAAS) are developed for miRNA selection in pan-cancer data. While the
developed MOHCNN helps in efficient patient classification, the STEAAS is use-
ful for identifying relevant miRNAs using explainable models. In MOHCNN,
dropout rate, choice of activation function, sizes of filters and pooling layer, and
the number of convolution layers, dense layers, and filters are optimized using
Bayesian optimization with tree parzen estimator in a multi-objective framework.
The objective functions are training error, validation error, and the number of
training parameters of the model. While the training and validation errors deter-
mine the accuracy of the model, the number of training parameters handles the
complexity of the model. In STEAAS, a set-theoretic explainable AI-based attri-
bution score is developed, using the Z −number concept, to identify the relevant
miRNAs. The score for each miRNA is an ordered pair. The first part, represent-
ing the class score of a miRNA, is computed by integrating BayLIME (Bayesian
Local Interpretable Model Agnostic Explanations) and SHAP (Shapley Additive
exPlanations) values. It is used to rank the miRNAs in each cancer class. The
second part in the ordered pair, indicating the reliability score of a miRNA for
belonging to a particular cancer class, is computed using the Gini index. The
performance of MOHCNN is observed to be superior to the related methods in
most of the cases. The results of MOHCNN in terms of accuracy, F1-score, and
MCC ranges from 0.99 to 1.00, 0.97 to 1.00, and 0.96 to 1.0, respectively. The
miRNAs selected by STEAAS are validated using OncomiR and ENCORI/Star-
base database and related literature. Most of the miRNAs selected by STEAAS
are mentioned as key biomarkers in various studies.

## Datasets
- <a href = "https://drive.google.com/file/d/1JEuFi3w0DlIRmUft3fjdHh8asn8hIu8J/view">Breast Dataset </a>
- <a href = "https://drive.google.com/file/d/1DMWWqt4dqb4Ixq57QAyMiVdhdI1zJ1Wx/view">Lung Dataset </a>
- <a href = "https://drive.google.com/file/d/1CR6wCbfdfqR3Dg7oZDtAy8kJPxD09RG4/view">Kidney Dataset </a>
- <a href = "https://drive.google.com/file/d/1JaNfq2m87z1KtuFZrNWRvS-AbkmXK_49/view">Classified Pan-cancer Sample (CPS) Dataset </a>

## Results



