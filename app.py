from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import cv2

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
ANNOTATED_FOLDER = 'static/annotated/'
PLANT_IMAGES_FOLDER = 'static/plant_images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANNOTATED_FOLDER'] = ANNOTATED_FOLDER
app.config['PLANT_IMAGES_FOLDER'] = PLANT_IMAGES_FOLDER

# Ensure the upload, annotated, and plant images directories exist
for folder in [UPLOAD_FOLDER, ANNOTATED_FOLDER, PLANT_IMAGES_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load the YOLO model
model = YOLO('model/best.pt')

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Perform prediction using the YOLO model
            results = model(file_path)

            # Save the annotated image
            annotated_image_path = os.path.join(app.config['ANNOTATED_FOLDER'], filename)
            annotated_img = results[0].plot()
            cv2.imwrite(annotated_image_path, annotated_img)

            # Extract the first prediction class name
            predictions = results[0].boxes  # This contains the bounding boxes and class predictions
            if len(predictions) > 0:
                # Get the class ID and map it to the class name
                predicted_class_id = int(predictions.cls[0].item())  # Get the first predicted class ID
                predicted_class_name = results[0].names[predicted_class_id]  # Map to class name

                # Path for the plant image
                plant_image_path = os.path.join(app.config['PLANT_IMAGES_FOLDER'], f"{predicted_class_name.lower()}.jpg")  # Assuming plant images are in .jpg format

                return render_template(
                    'index.html', 
                    class_name=predicted_class_name, 
                    uploaded_image=file_path, 
                    annotated_image=annotated_image_path,
                    plant_image=plant_image_path,
                    plant_name=predicted_class_name  # Pass the plant name
                )
            else:
                return 'No objects detected'

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
