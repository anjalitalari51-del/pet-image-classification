from PIL import Image
import os
import warnings

for folder in ["Cat", "Dog"]:
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with Image.open(filepath) as img:
                    img.load()

        except Exception as e:
            print("Bad image:", filepath)
            print("Error:", e)