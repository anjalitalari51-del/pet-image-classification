import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("pet_classifier.keras")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150,150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        label = "Dog"
    else:
        label = "Cat"

    return img, label

cat_img, cat_label = predict_image("Cat/1.jpg")
dog_img, dog_label = predict_image("Dog/187.jpg")

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(cat_img)
plt.title(f"Prediction: {cat_label}")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(dog_img)
plt.title(f"Prediction: {dog_label}")
plt.axis("off")

plt.show()