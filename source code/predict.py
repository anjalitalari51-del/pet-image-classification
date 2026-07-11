import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
# Load model
model = tf.keras.models.load_model("pet_classifier.keras")

# Image path
img_path = "test.jpg.png"   # change to your image

img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)
if prediction[0][0] > 0.5:
    label = "Cat"
else:
    label = "Dog"

plt.imshow(image.load_img(img_path))
plt.title(f"Prediction: {label}")
plt.axis("off")
plt.show()