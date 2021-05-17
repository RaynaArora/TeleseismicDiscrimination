# x_train, x_val, and x_test below are arrays of samples from the dataframes in this repository. They have been normalized to a range of -0.5 to 0.5. Dimensions are (6076, 1800, 1), (750, 3600, 1), and (782, 3600, 1).
# y_train, y_val, and y_test are arrays in which each element is 1 or 0 to represent that the corresponding sample is an explosion or earthquake, respectively.

import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
import matplotlib.pyplot as plt

import sklearn.metrics as metrics
from sklearn.metrics import roc_curve

def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)
    
    conv1 = keras.layers.Conv1D(filters=16, kernel_size=27, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    pool = keras.layers.MaxPooling1D()(conv1)
    
    bl1 = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(pool)
    bl1 = keras.layers.BatchNormalization()(bl1)
    bl1 = keras.layers.Dropout(0.2)(bl1)
    
    l1 = keras.layers.LSTM(64, return_sequences=False)(bl1)
    l1 = keras.layers.BatchNormalization()(l1)
    l1 = keras.layers.Dropout(0.1)(l1)
    
    d1 = keras.layers.Dense(64, activation="relu")(l1)
    d1 = keras.layers.BatchNormalization()(d1)
    d1 = keras.layers.Dropout(0.1)(d1)
        
    output_layer = keras.layers.Dense(1, activation="sigmoid")(d1)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=x_train.shape[1:])
#keras.utils.plot_model(model, show_shapes=True)

#Train the model
epochs = 60
batch_size = 32

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1,
    shuffle=True,
    validation_data=(x_val, y_val),
)

#Plot training and validation accuracy
metric = "accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()


# Evaluate model on test data

pred = model.predict(x_test, verbose=0)

t = 0.5
pred_pos = 0
truth_pos = 0
correct_pos = 0
for i in range (y_test.shape[0]):
    if y_test[i] == 1:
        truth_pos += 1
        if pred[i] >= t:
            correct_pos += 1
    
    if pred[i] >= t:
        pred_pos += 1

print ("Precision: " + str(correct_pos / pred_pos))
print ("Recall: " + str(correct_pos / truth_pos))

# ROC curve
fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
