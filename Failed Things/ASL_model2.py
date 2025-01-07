#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import coremltools as ct


# In[5]:


# MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# In[7]:


def extract_hand_landmarks(image):
    """Extract hand landmarks from an image using MediaPipe."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            return landmark_array.flatten()
    return np.zeros(21 * 3)


# In[13]:


def load_data(data_path):
    """Load data from a single dataset directory containing all gesture folders."""
    labels = []
    data = []
    
    print(f"Loading data from: {data_path}")
    
    # Get all gesture folders
    gesture_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    print(f"Found {len(gesture_folders)} gesture categories")
    
    # Process each gesture folder
    for gesture in gesture_folders:
        gesture_path = os.path.join(data_path, gesture)
        print(f"Processing gesture: {gesture}")
        
        # Get all images for this gesture
        images = [f for f in os.listdir(gesture_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_file in tqdm(images, desc=f"Processing {gesture}"):
            image_path = os.path.join(gesture_path, image_file)
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    hand_landmarks = extract_hand_landmarks(image)
                    data.append(hand_landmarks)
                    labels.append(gesture.lower())  # Convert to lowercase for consistency
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
        
        print(f"Processed {len(images)} images for {gesture}")
    
    return np.array(data), np.array(labels)


# In[17]:


# Load your ASL alphabet dataset (adjust path)
dataset_path = "./dataset/asl_alphabet_train/asl_alphabet_train/"

print("Starting data loading process...")
X, y = load_data(dataset_path)
print(f"Loaded {len(X)} total samples across {len(np.unique(y))} categories")


# In[19]:


# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Create label mapping
label_map = {label: idx for idx, label in enumerate(label_encoder.classes_)}
reverse_label_map = {idx: label for label, idx in label_map.items()}


# In[21]:


# Save label mappings for later use
import json
with open('label_mappings.json', 'w') as f:
    json.dump({
        'label_map': label_map,
        'reverse_label_map': {str(k): v for k, v in reverse_label_map.items()}  # Convert keys to strings for JSON
    }, f, indent=4)


# In[23]:


# Normalize the data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)


# In[25]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_categorical, test_size=0.2, random_state=42)


# In[27]:


# Reshape data for Conv1D
X_train = X_train.reshape(X_train.shape[0], 21, 3)
X_test = X_test.reshape(X_test.shape[0], 21, 3)


# In[29]:


# 7. Verify shapes
print("\nFinal shapes:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

# 8. Verify data ranges
print("\nData ranges:")
print(f"X_train min: {X_train.min()}, max: {X_train.max()}")
print(f"Sample label distribution: {y_encoded[:10]}")


# In[31]:


# Define the model
num_classes = len(label_encoder.classes_)
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(21, 3), padding='same'),
    MaxPooling1D(2),
    
    Conv1D(128, 3, activation='relu', padding='same'),
    MaxPooling1D(2),
    
    Conv1D(256, 3, activation='relu', padding='same'),
    GlobalAveragePooling1D(),
    
    Dense(128, activation='relu'),
    Dropout(0.5),
    
    Dense(64, activation='relu'),
    Dropout(0.3),
    
    Dense(num_classes, activation='softmax')
])


# In[33]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=150, validation_split=0.2, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# In[35]:


# Print model summary
model.summary()


# In[37]:


# Step 10: Save the model for edge deployment
model.save('asl_recognition_model811.h5')


# In[60]:


import joblib
joblib.dump(scaler, 'scaler.save')


# In[39]:


# Generate and plot confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

plt.figure(figsize=(20, 16))
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title('Confusion Matrix for Gesture Recognition', pad=20, size=16)
plt.xlabel('Predicted Label', labelpad=10)
plt.ylabel('True Label', labelpad=10)
plt.tight_layout()
plt.show()


# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()


# In[44]:


# Print classification report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, 
                          target_names=label_encoder.classes_,
                          zero_division=0))


# In[46]:


# Real-time prediction function
def real_time_prediction():
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        hand_landmarks = extract_hand_landmarks(frame)
        
        if np.any(hand_landmarks):
            # Prepare input data
            input_data = scaler.transform(hand_landmarks.reshape(1, -1))
            input_data = input_data.reshape(1, 21, 3)
            
            # Make prediction
            prediction = model.predict(input_data, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Get predicted label
            predicted_label = reverse_label_map[predicted_class]
            
            # Display prediction
            text = f"{predicted_label.upper()} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        cv2.imshow('Gesture Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# In[57]:


try:
    real_time_prediction()
except:
    print("Something else went wrong")
    cv2.destroyAllWindows()
finally:
    cv2.destroyAllWindows()


# In[49]:


cv2.destroyAllWindows()


# <h3>Converting models: </h3>

# In[12]:


from tensorflow.keras.models import load_model
import coremltools as ct


# In[14]:


model = load_model('asl_recognition_model811.h5')


# In[16]:


# Convert the model to Core ML format, specifying a fixed input shape
coreml_model = ct.convert(model, inputs=[ct.TensorType(shape=(1, 21, 3))])

# Save the model as an .mlmodel file
coreml_model.save('asl_recognition_model811.mlmodel')


# In[9]:


# Initialize the converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Perform the conversion
tflite_model = converter.convert()

# Save the converted model to a .tflite file
with open('asl_recognition_model811.tflite', 'wb') as f:
    f.write(tflite_model)

# Enable optimization (such as quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]


# In[ ]:




