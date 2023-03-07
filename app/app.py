from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        imagefile = request.files['imagefile']
        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=3000, debug=True)