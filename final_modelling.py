import glob
import os
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
from keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import features
MAX_FRAMES = 0
def load_npy(directory):
    global MAX_FRAMES
    npy_files = sorted(glob.glob(directory + "/*.npy"))
    MAX_FRAMES = max([np.load(npy).shape[1] for npy in npy_files])

    labels =[]
    data = []
    for mfcc_file in npy_files:
        mfcc_data = np.load(mfcc_file).astype(np.float32)
        mfcc_data = np.load(mfcc_file).astype(np.float32)
        mfcc_data = np.pad(mfcc_data, ((0, 0), (0, MAX_FRAMES - mfcc_data.shape[1])))
        data.append(mfcc_data)
        # Get label from parent directory name to avoid filename parsing issues
        label = Path(mfcc_file).stem[:-3]
        # print(label)
        labels.append(label)
    labels = np.array(labels)
    data = np.array(data, dtype=np.float32)
    # Normalize to approx [-1, 1] using global max-abs
    scale = max(abs(float(np.min(data))), float(np.max(data)))
    data = data / (scale + 1e-8)
    # Add channel dimension for Conv2D: (N, D, T, 1)
    data = data[..., np.newaxis]
    return data,labels

# Read class list; keep lower-case consistent
name_file = open("NAMES.txt")
names = [name.strip().lower() for name in name_file.readlines()]

# --------- Load & organize features ----------


# Unify the time dimension (number of frames)

data,labels = load_npy('data/features/audio')



# --------- Label encoding ----------
LE = LabelEncoder()
LE.fit(names)
labels = to_categorical(LE.transform(labels), num_classes=len(names))

# --------- Split data ----------
X_train, X_val, y_train, y_val = train_test_split(
    data, labels, test_size=0.2, random_state=0, stratify=labels
)

# --------- Build model ----------
def create_model():
    num_classes = len(names)
    D = data.shape[1]
    T = data.shape[2]
    model = Sequential()
    model.add(InputLayer(input_shape=(D, T, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu')) #64 try 32
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(512))  # originally 512, was overfitting
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

model = create_model()


TEST_NUM = 1
try:
    graph_dir = f"graphs/test{TEST_NUM}/"
    os.mkdir(graph_dir)
    print(f"Directory '{graph_dir}' created successfully.")
except FileExistsError:
    print(f"Directory '{graph_dir}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{graph_dir}'.")
except Exception as e:
    print(f"An error occurred: {e}")

# --------- Train / Load weights ----------
training = True
if training:
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])  #original 0.01
    model.summary()

    EPOCHS = 25 # original 25 try 10
    BATCH_SIZE = 16
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1
    )
    model.save_weights('model.weights.h5')

    # Training curves (plot acc/loss only)
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig(f"graphs/test{TEST_NUM}/train_accuracy.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig(f"graphs/test{TEST_NUM}/train_loss.png", dpi=200)
    plt.close()

if not training:
    model.load_weights('model.weights.h5')

# --------- Evaluation ----------
test_data, test_labels = load_npy('test/features')
test_labels = to_categorical(LE.transform(test_labels), num_classes=len(names))
# _train, X_test, _val, y_test = train_test_split(test_data, test_labels, test_size=0.8, random_state=0, stratify=test_labels)
predicted_probs = model.predict(test_data, verbose=0)
predicted = np.argmax(predicted_probs, axis=1)
actual = np.argmax(test_labels, axis=1)
accuracy = metrics.accuracy_score(actual, predicted)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Single-sample prediction example
predicted_prob = model.predict(test_data[0:1], verbose=0)
predicted_id = np.argmax(predicted_prob, axis=1)
predicted_class = LE.inverse_transform(predicted_id)
print("predicted:", predicted_class)
actual_label = test_labels[0]
actual_class = LE.inverse_transform([np.argmax(actual_label)])
print('actual:', actual_class)
# Confusion matrix (after evaluation)
confusion_matrix = metrics.confusion_matrix(actual, predicted, labels=range(len(names)))
confusion_matrix = confusion_matrix /np.max(confusion_matrix)
cm_display = ConfusionMatrixDisplay(confusion_matrix, display_labels=names)
cm_display.plot(include_values=True, xticks_rotation=90)
plt.tight_layout()
plt.savefig(f"graphs/test{TEST_NUM}/confusion_matrix.png", dpi=200)
plt.close()

'''
# Quick single-audio test
wav_path = 'data/test/test.wav'
mfcc = features.wav_to_mfcc(wav_path)
sd.play(sf.read(wav_path)[0], samplerate=16000)
sd.wait()
# Align to model input length
if mfcc.shape[1] < data.shape[2]:
    mfcc = np.pad(mfcc, ((0, 0), (0, data.shape[2] - mfcc.shape[1])))
else:
    mfcc = mfcc[:, :data.shape[2]]
mfcc = mfcc[np.newaxis, ..., np.newaxis]
p = model.predict(mfcc, verbose=0)
print("prediction: ", names[p.argmax()])
'''
