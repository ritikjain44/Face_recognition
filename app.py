import os
from flask import Flask, render_template, request, redirect, url_for
import csv
import cv2
import face_recognition
import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'dataset'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg'}

known_encodings = []
known_names = []
attendance_data = []


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def train_model():
    for student_dir in os.listdir(app.config['UPLOAD_FOLDER']):
        if os.path.isdir(os.path.join(app.config['UPLOAD_FOLDER'], student_dir)):
            student_images = []
            for image_file in os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], student_dir)):
                if allowed_file(image_file):
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], student_dir, image_file)
                    student_images.append(face_recognition.load_image_file(image_path))

            student_encodings = [face_recognition.face_encodings(img)[0] for img in student_images]
            known_encodings.extend(student_encodings)
            known_names.extend([student_dir] * len(student_encodings))


def save_attendance(student_name):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('attendance.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([student_name,current_time,"Present"])

def initialize_attendance():
    with open('attendance.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Student Name', 'Time', 'Attendance'])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        video_capture = cv2.VideoCapture(0)
        ret, frame = video_capture.read()

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                matched_index = matches.index(True)
                name = known_names[matched_index]
                save_attendance(name)

                return redirect(url_for('mark_attendance', student_name=name))

        video_capture.release()
        cv2.destroyAllWindows()

    return render_template('attendance.html')


@app.route('/show_attendance', methods=['GET', 'POST'])
def show_attendance():
    if request.method == 'POST':
        student_name = request.form['student_name']
        attendance_records = []
        with open('attendance.csv', mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                if row[0] == student_name:
                    attendance_records.append(row)

        return render_template('show_attendance.html', student_name=student_name, attendance_records=attendance_records)

    return render_template('show_attendance_input.html')

@app.route('/mark_attendance/<string:student_name>', methods=['GET', 'POST'])
def mark_attendance(student_name):
    if request.method == 'POST':
        save_attendance_csv()
        attendance_data.clear()
        return redirect(url_for('home'))

    return render_template('mark_attendance.html', student_name=student_name)


if __name__ == '__main__':
    initialize_attendance()
    train_model()
    app.run(debug=True, port=7001)
