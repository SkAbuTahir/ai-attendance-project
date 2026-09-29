
import dlib
import numpy as np
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st

from src.database.db import get_all_students


@st.cache_resource
def load_dlib_models():
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )
    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )
    return detector, sp, facerec


def get_face_embeddings(image_np):
    detector, sp, facerec = load_dlib_models()
    # upsample=2 improves detection of smaller/distant faces
    faces = detector(image_np, 2)
    encodings = []
    for face in faces:
        shape = sp(image_np, face)
        face_descriptor = facerec.compute_face_descriptor(image_np, shape, 1)
        encodings.append(np.array(face_descriptor))
    return encodings


@st.cache_resource
def get_trained_model():
    X, y = [], []

    student_db = get_all_students()
    if not student_db:
        return None

    for student in student_db:
        embedding = student.get('face_embedding')
        if embedding:
            X.append(np.array(embedding))
            y.append(student.get('student_id'))

    if len(X) == 0:
        return None

    # Need at least 2 classes for SVC; fall back to nearest-neighbour only
    clf = None
    if len(set(y)) >= 2:
        clf = SVC(kernel='linear', probability=True, class_weight='balanced')
        try:
            clf.fit(X, y)
        except ValueError:
            clf = None

    return {'clf': clf, 'X': X, 'y': y}


def train_classifier():
    st.cache_resource.clear()
    model_data = get_trained_model()
    return bool(model_data)


def _best_distance(X_train, y_train, student_id, query_encoding):
    """Return the minimum L2 distance across ALL embeddings for a student."""
    distances = [
        np.linalg.norm(np.array(X_train[i]) - query_encoding)
        for i, sid in enumerate(y_train)
        if sid == student_id
    ]
    return min(distances) if distances else float('inf')


def predict_attendance(class_image_np):
    encodings = get_face_embeddings(class_image_np)
    detected_student = {}

    model_data = get_trained_model()
    if not model_data:
        return detected_student, [], len(encodings)

    clf = model_data['clf']
    X_train = model_data['X']
    y_train = model_data['y']

    all_students = sorted(list(set(y_train)))
    DISTANCE_THRESHOLD = 0.55   # dlib recommended ≤0.6; tighter = fewer false positives
    CONFIDENCE_THRESHOLD = 0.45  # minimum SVC probability to accept a prediction

    for encoding in encodings:
        if clf is not None and len(all_students) >= 2:
            proba = clf.predict_proba([encoding])[0]
            best_idx = int(np.argmax(proba))
            best_conf = proba[best_idx]
            predicted_id = clf.classes_[best_idx]

            if best_conf < CONFIDENCE_THRESHOLD:
                continue
        else:
            predicted_id = all_students[0]

        dist = _best_distance(X_train, y_train, predicted_id, encoding)

        if dist <= DISTANCE_THRESHOLD:
            detected_student[predicted_id] = True

    return detected_student, all_students, len(encodings)
