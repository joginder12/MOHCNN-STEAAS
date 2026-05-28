
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
import shap
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import make_regression
#import np_utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from lime.lime_tabular import LimeTabularExplainer
from keras.regularizers import l2
from tensorflow.keras.layers import Conv1D, Dense,MaxPooling1D,Flatten, BatchNormalization
from keras.activations import relu, elu, linear, sigmoid
from sklearn.metrics import confusion_matrix 
from keras import backend as K
import os


input_files = [ 'I_Lung_Subtype20.csv']#, 'I_Lung_Subtype20.csv', 'I_Nor-Pan-cancer20.csv', 'I_Ntype_pan_can20.csv','I_Pan-can20.csv']    
# Define a function to process each input file

# Directory where you want to save the output files# 'I_grouped_n_tissue20.csv','I_breast_subtype20.csv',  
# Title: A Stacking Ensemble deep learning approach for cancer type classification based on TCGA data 
#Stacked_folder = 'jreg_test'

# Create the output folder if it doesn't exist
Stacked_folder = 'Lung_SHAP_explanation'

# Create the output folder if it doesn't exist
if not os.path.exists(Stacked_folder):
    os.makedirs(Stacked_folder)
# if not os.path.exists(Stacked_folder):
#     os.makedirs(Stacked_folder)
    
def process_input_file(input_file):
    # Read the input CSV file
    data = pd.read_csv(input_file, delimiter='\t')
    label_encoder = LabelEncoder().fit(data.miRNA_ID)
    labels = label_encoder.transform(data.miRNA_ID)
    classes = list(label_encoder.classes_)
    input = data.drop('miRNA_ID', axis=1)
    
    df = pd.read_csv("lung_explainability.csv", delimiter=',')
    print(df.head())
    data1 = df
    label_encoder1 = LabelEncoder().fit(data1.miRNA_ID)
    labels1 = label_encoder.transform(data1.miRNA_ID)
    classes1 = list(label_encoder1.classes_)
    input1 = data1.drop('miRNA_ID', axis=1)


    scaled_input1 = input1.values


    nb_features1 = scaled_input1.shape[1] # 1 is for column dimension
    nb_class1 = len(classes1)

    X_train_r1 = np.reshape(scaled_input1, (len(scaled_input1), nb_features1, 1))
    print(X_train_r1.shape)


    scaled_input = input.values


    nb_features = scaled_input.shape[1] # 1 is for column dimension
    nb_class = len(classes)
    X_train, X_test, y_train, y_test = train_test_split(scaled_input, labels, test_size=0.33, random_state=50)
    X_train_r = np.reshape(X_train, (len(X_train), nb_features, 1))
    X_test_r = np.reshape(X_test, (len(X_test), nb_features, 1))
    y_train_r = to_categorical(y_train, nb_class)
    y_test_r = to_categorical(y_test, nb_class)

    background= X_train_r[np.random.choice(X_train_r.shape[0], replace=False)]
#print(background)
    nn_in = X_train.shape[1]

    def network(nn_in):####new code to compute
        model = Sequential()
        model.add(Conv1D(filters = 33, kernel_size=3, input_shape=(nb_features, 1), activation='relu'))
        model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
       # model.add(layers.ReLU())
        model.add(MaxPooling1D(pool_size=5))
       # model.add(Dropout(0.13))

        model.add(Conv1D(filters = 18, kernel_size=5, input_shape=(nb_features, 1), activation='relu'))
        model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
       # model.add(layers.ReLU())
        model.add(MaxPooling1D(pool_size=4))
       # model.add(Dropout(0.19))

        # model.add(Conv1D(filters = 44, kernel_size=2, input_shape=(nb_features, 1)))
        # model.add(BatchNormalization(batch_size=128))
        # model.add(LeakyReLU())
        # model.add(MaxPooling1D(pool_size=3))
        # model.add(Dropout(0.10))
        model.add(Flatten())
        model.add(Dense(units=55)) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.17))
        model.add(Dense(units=68)) 
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(0.24))
        # model.add(Dense(units=51,activation='LeakyReLU'))
        # model.add(Dropout(0.43))
        model.add(Dense(units=nb_class,kernel_regularizer=l2(0.01),activation='softmax'))
#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])
        return model

    model=network(nn_in)
    
    nb_epoch = 50
    cvscores = []
    cvscores1 = []
    confusion_matrices = []
    sss = StratifiedKFold(n_splits=10, shuffle=True, random_state=50)
    i=1
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    for input_index, valid_index in sss.split(scaled_input, labels):
        print("\nRound: %d\n"% i)
        i=i+1
        X_input, X_valid = scaled_input[input_index], scaled_input[valid_index]
        y_input, y_valid = labels[input_index], labels[valid_index]
# reshape train data
        X_input_r = np.reshape(X_input, (len(X_input), nb_features, 1))
    # reshape validation data
        X_valid_r = np.reshape(X_valid, (len(X_valid), nb_features, 1))
        y_input_r = to_categorical(y_input, nb_class)
        y_valid_r = to_categorical(y_valid, nb_class)
        history=model.fit(X_input_r, y_input_r, epochs=nb_epoch, validation_data=(X_valid_r, y_valid_r),callbacks=[callback])
        scores = model.evaluate(X_valid_r, y_valid_r, verbose=0)
        print(len(scores))
        # print("\n%s: %.2f%% \n" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        cvscores1.append(scores[0] * 100)
        
        
        y_predict=model.predict(X_valid_r, batch_size=10, verbose=0)
        y_valid_r = np.argmax(y_valid_r, axis=1)
        y_predict= np.argmax(y_predict, axis=1)
        #print(y_valid_r, y_predict)
        confusion_matrices.append(confusion_matrix(y_valid_r, y_predict))
#print('Confusion Matrix\n')
#print(confusion)
#print(cvscores1)
# print("Training accuracy: ", history.history['accuracy'])
     #   hist_df = pd.DataFrame(history.history)
     #   output_file = f"{os.path.splitext(input_file)[0]}_stacked_{i-1}.csv"
     #   output_path = os.path.join(Stacked_folder, output_file)
     #   hist_df.to_csv(output_path, index=False)
#
      #  print(f'{input_file} processed and saved as {output_file}')

    del X_input, X_valid, y_input, y_valid, X_input_r, X_valid_r, y_input_r, y_valid_r, scores
    
    predictions = model.predict(X_train_r)
    print(predictions[:10])

   # Initialize the LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X_train_r1.reshape(X_train_r1.shape[0], -1),  # Flatten for tabular explainer
        mode='classification',
        discretize_continuous=False
    )

    # Prediction function adjusted for binary classification
    #print(x.reshape(x.shape[0], 10, 1))

    def predict_fn(x):
        preds = model.predict(x.reshape(x.shape[0], nb_features, 1))
        return np.hstack([1 - preds, preds])  # Convert to [prob_class_0, prob_class_1]

    # Choose a sample to explain
    #i = 20  # Index of the sample to explain
    df21_importance = pd.DataFrame()
    df21_feature = pd.DataFrame()
    
    
    for i in range(len(X_train_r1)):
        sample = X_train_r1[i].reshape(1, -1)

    # Generate explanation
        explanation = explainer.explain_instance(
        data_row=sample.flatten(),
        predict_fn=predict_fn,
        model_regressor=BayesianRidge(max_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True, copy_X=True, verbose=False),
        num_features=nb_features
        )

    # Show explanation
        explanation.show_in_notebook(show_table=True, show_all=False)

       # print(explanation)
    # Get a list of (feature, importance) pairs
        explanation_list = explanation.as_list()
        explanation_list_sorted = sorted(explanation_list, key=lambda x: int(x[0]))

    # Print each feature and its importance
       # print("LIME Explanation:")
       # for feature, importance in explanation_list:
            #print(f"Feature: {feature}, Importance: {importance}")
    
        importance_values = [item[0] for item in explanation_list]
        importance_values1 = [item[1] for item in explanation_list]
        
        print(importance_values)
        print(importance_values1)

# Create a DataFrame with the importance values
        df21_importance[i] = pd.DataFrame(importance_values, columns=["Importance_{i}"])
        df21_feature[i] = pd.DataFrame(importance_values1, columns=["Feature_{i}"])

    # Print the explanation as a string
      #  print("\nFull Explanation:")
       # print(explanation_list)
    df21_importance.to_csv("importance_Lung.csv", index=False)
    df21_feature.to_csv("feature_Lung.csv", index=False)
    explainer = shap.GradientExplainer(model, X_train_r)
    shap_values = explainer(X_train_r1).values
    print(shap_values[0])
#  print(shap_values[1])
  # print(shap_values[0].shape)
    a, b, c=shap_values[0].shape
    for i in range(len(shap_values)):
      reshaped_2d_array = shap_values[i].reshape(a, c)
      hist_df = pd.DataFrame(reshaped_2d_array)
      output_file = f"{os.path.splitext(input_file)[0]}_reshape_{i}.csv"
      output_path = os.path.join(Stacked_folder, output_file)
      hist_df.to_csv(output_path, index=False)
      print(f'{input_file} processed and saved as {output_file}')
  


for input_file in input_files:
    process_input_file(input_file)

print('All files processed and saved.')
