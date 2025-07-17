#I used 20250527_tr01_top, 20250613_tr05_top and 20250613_tr05_face
video_data = {
    "video1":{
        "path":r'C:\Yasmine\Motion_intention_detection\Lilo_20250527_tr01.top.avi', 
        "intention_intervals":[(372,451),(893,949), (1398,1449), (1464,1527), (1856,1915)],
        "no_intention_intervals":[(1,371), (480,892), (963,1397), (1459,1463), (1579,1855), (1925,2000)]
    },
    "video2":{
        "path":r'C:\Yasmine\Motion_intention_detection\Lilo_20250613_tr05.face.mp4', 
        "intention_intervals":[(29,58),(365,457), (679,738)],
        "no_intention_intervals":[(1,28), (81,364), (483,670), (752,1001)]
    },
    "video3":{
        "path":r'C:\Yasmine\Motion_intention_detection\Lilo_20250613_tr05.top.mp4', 
        "intention_intervals":[(29,58),(365,457), (679,738)],
        "no_intention_intervals":[(1,28), (81,364), (483,670), (752,1001)]
    },
}

max_frames = 6000

# Fonction features

'''def extract_face_embedding(frame):
    face_locations = face_recognition.face_locations(frame)
    if len(face_locations) == 0:
        return np.zeros(128)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    return face_encodings[0]  # vecteur 128D'''

def extract_features_from_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Génération labels
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

# Split train/test
split_point = int(0.75 * len(X))
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = Y[:split_point], Y[split_point:]
frame_indices_test = frame_indices[split_point:]

# Classifieurs à tester
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "SVM": SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB()
}

results = {}

# Entraînement et prédiction pour chaque modèle
for name, clf in models.items():
    print(f"\n=== {name} ===")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    results[name] = {
        'y_pred': y_pred,
        'cm': cm,
        'report': report
    }
    print(report)


#Confusion matric
fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4))
if len(models) == 1:
    axes = [axes]  # pour compatibilité

for ax, (name, res) in zip(axes, results.items()):
    sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Intention', 'Intention'], 
                yticklabels=['No Intention', 'Intention'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(name)

plt.tight_layout()
plt.show()

# Courbes temporelles
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
