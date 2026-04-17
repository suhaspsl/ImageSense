from flask import Flask, request, jsonify,render_template
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model/image_model.h5")

# Class labels (EDIT THIS)
class_names = ["Cat", "Dog"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]

        img = Image.open(file).resize((224, 224))  # adjust size to your model
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        predicted_class = class_names[np.argmax(prediction)]

        return jsonify({"prediction": predicted_class})

    except Exception as e:
        print(e)  # VERY IMPORTANT (check terminal)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)



    