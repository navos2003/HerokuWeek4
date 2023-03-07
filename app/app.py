import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from keras.applications.mobilenet import decode_predictions
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input

UPLOAD_FOLDER = 'images'
saved_model = 'model/data.h5'
if os.path.exists(saved_model):
    model = load_model(saved_model)
else:
    model = MobileNet()
    model.save(saved_model)

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
#@app.route('/', methods=['GET', 'POST'])
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['imagefile']
    filename = secure_filename(file.filename)

    if file is None or file.filename == "":
        out_message = "No file"
    elif not allowed_file(file.filename):
        out_message = "Format not currently supported"
    else:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace("\\","/")
        file.save(image_path)
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        try:
            y_pred = model.predict(image)
            label = decode_predictions(y_pred)
            label = label[0][0]
            word = [word.replace('_', ' ') for word in label[1]]
            result = ''.join(word)
            if(label[1][0] in ['a','e','i','o','u']):
                out_message = "Is your image an %s?" %(result)
            else:
                out_message = "Is your image a %s?" %(result)
        except:
            out_message = "Error occured"

    return render_template('index.html', out_message=out_message)

ALLOWED = {"jpg", "png", "jpeg"}
def allowed_file(filename):
    out = ("." in filename) and (filename.rsplit(".", 1)[1].lower() in ALLOWED)
    return out

if __name__ == '__main__':
    app.run(port=4000,debug=True)