import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

video_data = {
    "video1": {
        "path": r'C:\Yasmine\Motion_intention_detection\Lilo_20250527_tr01.top.avi',
        "intention_intervals": [(372, 451), (893, 949), (1398, 1449), (1464, 1527), (1856, 1915)],
        "no_intention_intervals": [(1, 371), (480, 892), (963, 1397), (1459, 1463), (1579, 1855), (1925, 2000)],
        "max_frames": 2000
    },
    "video2": {
        "path": r'C:\Yasmine\Motion_intention_detection\Lilo_20250613_tr05.face.mp4',
        "intention_intervals": [(29, 58), (365, 457), (679, 738)],
        "no_intention_intervals": [(1, 28), (81, 364), (483, 670), (752, 1001)],
        "max_frames": 1000
    },
    "video3": {
        "path": r'C:\Yasmine\Motion_intention_detection\Lilo_20250613_tr05.top.mp4',
        "intention_intervals": [(29, 58), (365, 457), (679, 738)],
        "no_intention_intervals": [(1, 28), (81, 364), (483, 670), (752, 1001)],
        "max_frames": 1000
    }
}

def extract_features_from_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def process_video(video_path, intention_intervals, no_intention_intervals, max_frames=2000):
    frames_to_keep = {}
    for start, end in intention_intervals:
        for idx in range(start, end + 1):
            frames_to_keep[idx] = 1
    for start, end in no_intention_intervals:
        for idx in range(start, end + 1):
            frames_to_keep[idx] = 0

    cap = cv2.VideoCapture(video_path)
    X, Y, frame_indices = [], [], []
    frame_idx = 1
    while cap.isOpened() and frame_idx <= max_frames:
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
    return np.array(X), np.array(Y), np.array(frame_indices)

X_total, Y_total, indices_total = [], [], []

for video_name, data in video_data.items():
    print(f"[INFO] Processing {video_name}")
    X, Y, indices = process_video(
        data["path"],
        data["intention_intervals"],
        data["no_intention_intervals"],
        data["max_frames"]
    )
    X_total.append(X)
    Y_total.append(Y)
    indices_total.append(indices)

X = np.vstack(X_total)
Y = np.hstack(Y_total)
frame_indices = np.hstack(indices_total)

print(f"[INFO] Features shape: {X.shape}, Labels shape: {Y.shape}")
print(f"[INFO] Label distribution: {np.bincount(Y)}")

split_point = int(0.75 * len(X))
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = Y[:split_point], Y[split_point:]
frame_indices_test = frame_indices[split_point:]

X_train_0 = X_train[y_train == 0]
X_train_1 = X_train[y_train == 1]
y_train_1 = np.ones(len(X_train_1))
X_train_0_down = resample(X_train_0, replace=False, n_samples=len(X_train_1), random_state=42)
y_train_0_down = np.zeros(len(X_train_0_down))

X_train_bal = np.vstack([X_train_1, X_train_0_down])
y_train_bal = np.hstack([y_train_1, y_train_0_down])

print(f"[INFO] Balanced training set: {np.bincount(y_train_bal.astype(int))}")

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "SVM": SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB()
}

results = {}

for name, clf in models.items():
    print(f"\n=== {name} ===")
    clf.fit(X_train_bal, y_train_bal)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    results[name] = {
        'y_pred': y_pred,
        'cm': cm,
        'report': report
    }
    print(report)

fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4))
if len(models) == 1:
    axes = [axes]
for ax, (name, res) in zip(axes, results.items()):
    sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Intention', 'Intention'],
                yticklabels=['No Intention', 'Intention'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(name)
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))
plt.fill_between(frame_indices_test, 0, y_test, step='post', color='blue', alpha=0.2, label='Ground Truth')
plt.plot(frame_indices_test, y_test, drawstyle='steps-post', color='blue', label='Ground Truth')

colors = ['orange', 'green', 'red', 'purple']
for (name, res), color in zip(results.items(), colors):
    plt.plot(frame_indices_test, res['y_pred'], drawstyle='steps-post', alpha=0.7, label=f'Prediction: {name}', color=color)

plt.xlabel('Frame Index')
plt.ylabel('Class')
plt.title('Ground Truth vs Predictions (all models)')
plt.legend()
plt.tight_layout()
plt.show()
