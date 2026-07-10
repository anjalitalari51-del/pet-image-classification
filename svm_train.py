import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = []
labels = []

categories = ["Cat", "Dog"]

for category in categories:
    path = category
    label = categories.index(category)

    for img_name in os.listdir(path)[:500]:
        try:
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)

            if img is not None:
                img = cv2.resize(img, (64, 64))
                img = img.flatten()

                data.append(img)
                labels.append(label)

        except Exception as e:
            pass

data = np.array(data)
labels = np.array(labels)

print("Dataset Loaded:", len(data), "images")

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

print("Training SVM Model...")

model = SVC(kernel="linear")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Done")