import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV file containing reviews and sentiments
df = pd.read_csv('Final_Review_Dataset_Copy.csv')
df1 = pd.read_csv('Pos_Neg_Final.csv')

reviews = df['Summary_Tamil'].values
sentiments = df1['Sentiment_1'].values

reviews = [str(review) for review in reviews]
#author dharshan,nandheeswaran,hemanth
vocab_size = 5000
max_sequence_length = 200
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
x = pad_sequences(sequences)

num_classes = 3
y = tf.keras.utils.to_categorical(sentiments, num_classes=num_classes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#accuracy for bilstm 0.93
model = tf.keras.Sequential([
    Embedding(input_dim=vocab_size, output_dim=32),
    Bidirectional(LSTM(units=64, activation='tanh', return_sequences=True)),
    Bidirectional(LSTM(units=64, activation='tanh')),
    Dense(units=num_classes, activation='softmax')
])
#author dharshan,nandheeswaran,hemanth
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

y_pred = model.predict(x_test)

y_true = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(y_true, y_pred_classes))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

# Plotting the training and validation loss
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Training and Testing Loss by Epoch')
plt.xlabel('Epoch')
plt.ylabel('Categorical Crossentropy Loss')
plt.legend(loc='upper right')
plt.show()

#author dharshan,nandheeswaran,hemanth
auc_scores = []
for i in range(num_classes):
    auc_scores.append(roc_auc_score(y_test[:, i], y_pred[:, i]))
    print(f"AUC for class {i}: {auc_scores[i]:.4f}")

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = ['blue', 'red', 'green']
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=3, label=f'ROC curve (class {i}, area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve')
plt.legend(loc="lower right")
plt.show()
