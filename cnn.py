import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV file containing reviews and sentiments
df = pd.read_csv('Final_Review_Dataset_Copy.csv')
df1 = pd.read_csv('Pos_Neg_Final.csv')

reviews = df['Summary_Tamil'].values
sentiments = df1['Sentiment_1'].values

reviews = [str(review) for review in reviews]

#using 3 classes
vocab_size = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
x = pad_sequences(sequences)

num_classes = 3
y = tf.keras.utils.to_categorical(sentiments, num_classes=num_classes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    Embedding(input_dim=vocab_size, output_dim=32),
    Conv1D(64, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(units=num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(np.argmax(y_test, axis=1), y_pred_classes))

# Binarize the labels
lb = LabelBinarizer()
lb.fit(y_test)

y_test_binarized = lb.transform(np.argmax(y_test, axis=1))
y_pred_binarized = lb.transform(y_pred_classes)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_binarized[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
#author nandheeswaran,dharshan,hemanth
# Plot the ROC curve
plt.figure()
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Calculate AUC for each class
auc_scores = []
for i in range(num_classes):
    auc_scores.append(roc_auc_score(y_test[:, i], y_pred[:, i]))
    print(f"AUC for class {i}: {auc_scores[i]:.4f}")

# Confusion Matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

#author nandheeswaran,dharshan,hemanth
# Training and Testing Loss
plt.figure(figsize=(10,7))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
