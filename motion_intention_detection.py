import cv2
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


video_path = r'U:\Yasmine\Project\20250527\Lilo_20250527_tr01_face.avi'

intention_intervals = [(372,451),(893,949), (1398,1449), (1464,1527), (1856,1915)]
no_intention_intervals = [(1,371), (480,892), (963,1397), (1459,1463), (1579,1855), (1925,2000)]

frame_downsample = 1 #try a higher value if it's very slow

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Total frames in the video : {total_frames}")

y = np.zeros(total_frames, dtype=int)
for start, end in intention_intervals:
    y[start:end+1] = 1


#A changer pour récup des features plus compliquées si besoin
def extract_features_from_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

X = []
Y = []
frame_indices = []

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % frame_downsample == 0:
        features = extract_features_from_frame(frame)
        X.append(features)
        Y.append(y[frame_idx])
        frame_indices.append(frame_idx)
    frame_idx += 1

cap.release()
X = np.array(X)
Y = np.array(Y)
print(f"[INFO] Features shape: {X.shape}, Labels shape: {Y.shape}")

# Training set and testing set 
split_point = int(len(X) * 0.75)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = Y[:split_point], Y[split_point:]
indices_test = frame_indices[split_point:]

# learning > random forest, try another classifier if needed 
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("[INFO] Classification report :")
print(classification_report(y_test, y_pred))

def extract_predicted_intervals(predictions, indices):
    intervals = []
    in_interval = False
    for idx, (frame_idx, pred) in enumerate(zip(indices, predictions)):
        if pred == 1 and not in_interval:
            start = frame_idx
            in_interval = True
        elif pred == 0 and in_interval:
            end = frame_idx - 1
            intervals.append((start, end))
            in_interval = False
    if in_interval:
        intervals.append((start, indices[-1]))
    return intervals

predicted_intervals = extract_predicted_intervals(y_pred, indices_test)

print("[INFO] Intervalles prédits d'intention motrice :")
for start, end in predicted_intervals:
    print(f"De {start} à {end}")


plt.figure(figsize=(15, 3))
plt.plot(indices_test, y_test, label="Ground Truth", drawstyle='steps-post')
plt.plot(indices_test, y_pred, label="Prediction", drawstyle='steps-post', alpha=0.7)
plt.xlabel("Frame")
plt.ylabel("Intention (1) / Pas d'intention (0)")
plt.title("Détection d'intention motrice frame par frame")
plt.legend()
plt.tight_layout()
plt.show()
