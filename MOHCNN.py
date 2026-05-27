import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
#import shap
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#import np_utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from keras.regularizers import l2
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
from tensorflow.keras.layers import Conv1D, Dense,MaxPooling1D,Flatten, BatchNormalization
from sklearn.metrics import confusion_matrix
import tensorflow.keras.backend as K
import os
import matplotlib.pyplot as plt
import seaborn as sns
import csv 
#shap.initjs()

# List of input CSV file names
#input_files = ['I_breast_subtype20.csv','I_breast_subtype30.csv','I_breast_subtype40.csv','I_breast_subtype50.csv', 'I_grouped_n_tissue20.csv','I_grouped_n_tissue30.csv','I_grouped_n_tissue40.csv','I_grouped_n_tissue50.csv', 'I_Kidney_subtype20.csv','I_Kidney_subtype30.csv','I_Kidney_subtype40.csv','I_Kidney_subtype50.csv','I_Lung_Subtype20.csv','I_Lung_Subtype30.csv','I_Lung_Subtype40.csv','I_Lung_Subtype50.csv', 'I_Nor-Pan-cancer20.csv','I_Nor-Pan-cancer30.csv','I_Nor-Pan-cancer40.csv','I_Nor-Pan-cancer50.csv', 'I_Ntype_pan_can20.csv', 'I_Ntype_pan_can30.csv', 'I_Ntype_pan_can40.csv', 'I_Ntype_pan_can50.csv',  'I_Pan-can20.csv', 'I_Pan-can30.csv','I_Pan-can40.csv','I_Pan-can50.csv']
input_files = ['I_Pan-can20.csv']#,  'I_grouped_n_tissue20.csv', 'I_Kidney_subtype20.csv', 'I_Lung_Subtype20.csv', 'I_Nor-Pan-cancer20.csv', 'I_Ntype_pan_can20.csv','I_Pan-can20.csv']
# Define a function to process each input file

# Directory where you want to save the output files#
# Title: A Stacking Ensemble deep learning approach for cancer type classification based on TCGA data
#Stacked_folder = 'jreg_test'

# Create the output folder if it doesn't exist
# if not os.path.exists(Stacked_folder):
#     os.makedirs(Stacked_folder)

def process_input_file(input_file):
    # Read the input CSV file
    data = pd.read_csv(input_file, delimiter='\t')
    label_encoder = LabelEncoder().fit(data.miRNA_ID)
    labels = label_encoder.transform(data.miRNA_ID)
    classes = list(label_encoder.classes_)
    input = data.drop('miRNA_ID', axis=1)


    scaled_input = input.values
    noise_level_array = [0.05, 0.10, 0.15, 0.2, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5]
    for i in noise_level_array:
        noise_level = i # 20% noise
        csv_file = f"CPS_noise{i}".replace('.', '_') + ".csv"
        noise = np.random.normal(0, noise_level * scaled_input.std(), scaled_input.shape)
        scaled_valuess = scaled_input + noise

        nb_features = scaled_input.shape[1] # 1 is for column dimension
        nb_class = len(classes)
        X_train, X_test, y_train, y_test = train_test_split(scaled_valuess, labels, test_size=0.33, random_state=50)
        X_train_r = np.reshape(X_train, (len(X_train), nb_features, 1))
        X_test_r = np.reshape(X_test, (len(X_test), nb_features, 1))
        y_train_r = to_categorical(y_train, nb_class)
        y_test_r = to_categorical(y_test, nb_class)

        background= X_train_r[np.random.choice(X_train_r.shape[0], replace=False)]
#print(background)
        nn_in = X_train.shape[1]
        def recall_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        def f1_m(y_true, y_pred):
            precision = precision_m(y_true, y_pred)
            recall = recall_m(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))

        def mcc(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
            false_positives = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
            false_negatives = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

            numerator = (true_positives * true_negatives - false_positives * false_negatives)
            denominator = K.sqrt((true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives + false_positives) * (true_negatives + false_negatives))

            return numerator / (denominator + K.epsilon())

        def network(nn_in):####new code to compute
            model = Sequential()
        # model.add(Conv1D(filters = 33, kernel_size=3, input_shape=(nb_features, 1)))
        # model.add(BatchNormalization(batch_size=128))
        # model.add(layers.ReLU())
        # model.add(MaxPooling1D(pool_size=5))
       # model.add(Dropout(0.13))

            model.add(Conv1D(filters = 18, kernel_size=5, input_shape=(nb_features, 1)))
            model.add(BatchNormalization(batch_size=128))
            model.add(layers.ReLU())
            model.add(MaxPooling1D(pool_size=4))
       # model.add(Dropout(0.19))

            model.add(Conv1D(filters = 44, kernel_size=2, input_shape=(nb_features, 1)))
            model.add(BatchNormalization(batch_size=128))
            model.add(layers.ReLU())
            model.add(MaxPooling1D(pool_size=3))
            model.add(Dropout(0.10))
            model.add(Flatten())
            model.add(Dense(units=55,activation='LeakyReLU'))        
            model.add(Dropout(0.17))
            model.add(Dense(units=68,activation='LeakyReLU'))        
            model.add(Dropout(0.24))
        # model.add(Dense(units=51,activation='LeakyReLU'))
        # model.add(Dropout(0.43))
            model.add(Dense(units=nb_class,kernel_regularizer=l2(0.01),activation='softmax'))
#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=['accuracy',f1_m,precision_m, recall_m, mcc])
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
            start_train = time.time()
            history=model.fit(X_input_r, y_input_r, epochs=nb_epoch, validation_data=(X_valid_r, y_valid_r), batch_size=128, callbacks=[callback])
            scores = model.evaluate(X_valid_r, y_valid_r, verbose=0)
            train_time = time.time() - start_train
            print(len(scores))
            print(train_time)
        # print("\n%s: %.2f%% \n" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
            cvscores1.append(scores[0] * 100)




     #   print(f'{input_file} processed and saved as {output_file}')

        del X_input, X_valid, y_input, y_valid, X_input_r, X_valid_r, y_input_r, y_valid_r, scores
        start_train1 = time.time()
        y_predict=model.predict(X_test_r, batch_size=10, verbose=0)
        y_pred = np.argmax(y_predict, axis=1)
        test_time = time.time() - start_train1
        print(test_time)

# Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        mm = matthews_corrcoef(y_test, y_pred)

        print("\n📊 Test Performance Metrics:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"MM (MCC):  {mm:.4f}")
        
        
# Open the CSV file in write mode
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)

    # Write the header row
            writer.writerow(["Metric", "Value"])

    # Write the data rows
            writer.writerow(["Accuracy", acc])
            writer.writerow(["Precision", prec])
            writer.writerow(["Recall", rec])
            writer.writerow(["F1-score", f1])
            writer.writerow(["MM (MCC)", mm])

        print(f"\nPerformance metrics saved to {csv_file}")

# -------------------------------
# 5. Plot Confusion Matrix
# -------------------------------
   #  cm = confusion_matrix(y_test, y_pred)
   #  plt.figure(figsize=(10, 10))
   #  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
   #  #sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues',
   #  #        xticklabels=class_names, yticklabels=class_names,
   #  #        cbar=False)
   # # plt.title('Confusion Matrix - Test Data (Multiclass)')
   #  plt.xlabel('Predicted Class Label')
   #  plt.ylabel('Actual Class Label')
   #  plt.show()


for input_file in input_files:
    process_input_file(input_file)

print('All files processed and saved.')