import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

video_path = r'C:\Users\daoui\Documents\Yasmine\Stage NR 2025\Projet\Lilo_20250527_tr01.face.avi'

import os

print(f"[INFO] Checking existence of: {video_path}")
print(f"[INFO] Exists: {os.path.exists(video_path)}")
intention_intervals = [(372,451),(893,949), (1398,1449), (1464,1527), (1856,1915)]
no_intention_intervals = [(1,371), (480,892), (963,1397), (1459,1463), (1579,1855), (1925,2000)]

max_frames = 2000


cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("[ERROR] Check the problem")
else:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Total frames according to OpenCV: {total_frames}")

cap.release()

#extraction features > voir si autres features plus compliquées
def extract_features_from_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[INFO] Total frames in video: {total_frames}")
cap.release()
#SET
frames_to_keep = {}
for start, end in intention_intervals:
    for idx in range(start, end + 1):
        frames_to_keep[idx] = 1  # label 1
for start, end in no_intention_intervals:
    for idx in range(start, end + 1):
        frames_to_keep[idx] = 0  # label 0

cap = cv2.VideoCapture(video_path)
X = []
Y = []
frame_indices = []

frame_idx = 1
while cap.isOpened() and frame_idx < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx in frames_to_keep:
        features = extract_features_from_frame(frame)
        X.append(features)
        Y.append(frames_to_keep[frame_idx])
        frame_indices.append(frame_idx)
    frame_idx += 1

cap.release()
X = np.array(X)
Y = np.array(Y)
frame_indices = np.array(frame_indices)

print(f"[INFO] Features shape: {X.shape}, Labels shape: {Y.shape}")
print(f"[INFO] Label distribution: {np.bincount(Y)}")
# Séparation
split_point = int(0.75 * len(X))
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = Y[:split_point], Y[split_point:]
frame_indices_test = frame_indices[split_point:]
#Learning
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)
# Prediction
y_pred = clf.predict(X_test)
#Results
print("[INFO] Classification Report:")
print(classification_report(y_test, y_pred))

print("[INFO] Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#figure
# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Intention', 'Intention'], yticklabels=['No Intention', 'Intention'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Courbes plus claires
plt.figure(figsize=(15, 4))
plt.fill_between(frame_indices_test, 0, y_test, step='post', color='blue', alpha=0.3, label='Ground Truth')
plt.fill_between(frame_indices_test, 0, y_pred, step='post', color='orange', alpha=0.3, label='Prediction')
plt.plot(frame_indices_test, y_test, drawstyle='steps-post', color='blue')
plt.plot(frame_indices_test, y_pred, drawstyle='steps-post', color='orange')
plt.xlabel('Frame Index')
plt.ylabel('Class')
plt.title('Ground Truth vs Prediction over Frames')
plt.legend()
plt.tight_layout()
plt.show()
