from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
)
import os
from werkzeug.utils import secure_filename
from modeltesting import evaluateresume

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
app.config["GENERATED_FOLDER"] = "generated/"  # Folder to store generated images
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload limit

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["GENERATED_FOLDER"], exist_ok=True)


# @app.route("/", methods=["GET"])
# def render_index():
#     return render_template("index.html")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_description = request.form["job_description"]
        file = request.files["resume"]
        if file.filename == "":
            return redirect(request.url)

        filename = secure_filename(file.filename)
        resumepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(resumepath)

        result_image_path, score = evaluateresume(job_description, resumepath)
        result_image_url = url_for(
            "generated_image", filename=os.path.basename(result_image_path)
        )

        return render_template("index.html", result=result_image_url, score=score)
    return render_template("index.html")


@app.route("/generated/<filename>")
def generated_image(filename):
    return send_from_directory(app.config["GENERATED_FOLDER"], filename)


@app.route("/result/<filename>")
def result(filename):
    return send_from_directory("static/images", filename)


if __name__ == "__main__":
    app.run(debug=True)
