import cv2
import dlib
import numpy as np

# Load the pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load a sample image of a known person
known_image = dlib.load_rgb_image("known_person.jpg")

# Detect faces in the known image
known_faces = detector(known_image)

# Extract face encodings of the known faces
known_encodings = [face_recognizer.compute_face_descriptor(known_image, shape_predictor(known_image, face)) for face in known_faces]

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Iterate over the detected faces
    for face in faces:
        # Predict the face encodings of the current face
        face_encoding = face_recognizer.compute_face_descriptor(frame, shape_predictor(frame, face))

        # Convert face encodings to numpy arrays for subtraction
        known_encodings_np = np.array(known_encodings)

        # Compare the face encoding with the known encodings
        distances = np.linalg.norm(face_encoding - known_encodings_np, axis=1)
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]

        # Threshold for face recognition (you can adjust this according to your needs)
        if min_distance < 0.6:
            # Recognized face
            recognized_name = "Known Person"
        else:
            # Unknown face
            recognized_name = "Unknown"

        # Draw a rectangle around the face and display the recognized name
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
video_capture.release()
cv2.destroyAllWindows()
