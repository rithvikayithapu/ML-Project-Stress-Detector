from Pipeline import *

import numpy as np
import cv2
import pickle
import logging
import time

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.ERROR, format="[%(levelname)s] [{}] %(message)s".format(time.time()))

class LoadModels(Stage):
    def __init__(self):
        super(LoadModels, self).__init__()
    
    def execute(self, io, static_io):
        face_model_xml_file = static_io.model_files['face_detection_model']
        emotion_model_json_file, emotion_model_weights_file = static_io.model_files['emotion_detection_model']
        strees_classifier_model_sav_file = static_io.model_files['stress_classifier_model']

        try:
            face_detection_model = cv2.CascadeClassifier(face_model_xml_file)
            with open(emotion_model_json_file, 'r') as json_file:
                emotion_detection_model = model_from_json(json_file.read())
                emotion_detection_model.load_weights(emotion_model_weights_file)
            stress_classifier_model = pickle.load(open(strees_classifier_model_sav_file, 'rb'))
        except Exception as e:
            logging.fatal(e)
            return status.FAILURE

        static_io.models = dict()
        static_io.models['face_detection_model'] = face_detection_model
        static_io.models['emotion_detection_model'] = emotion_detection_model
        static_io.models['stress_classifier_model'] = stress_classifier_model

        return status.SUCCESS

class GrabFrame(Stage):
    def __init__(self):
        super(GrabFrame, self).__init__(input_keys=['frame'], output_keys=['gray_frame'])
    
    def execute(self, io, static_io):
        try:
            io.gray_frame = cv2.cvtColor(io.frame, cv2.COLOR_BGR2GRAY)
            return status.SUCCESS
        except Exception as e:
            logging.error(e)
            return status.FAILURE

class FaceDetection(Stage):
    def __init__(self):
        super(FaceDetection, self).__init__(input_keys=['gray_frame', 'models'], output_keys=['faces'])
    
    def execute(self, io, static_io):
        try:
            faces = static_io.models['face_detection_model'].detectMultiScale(io.gray_frame)
        except Exception as e:
            logging.error(e)
            return status.FAILURE
        
        if len(faces) == 0:
            logging.warn('No faces detected. Terminating this pipeline')
            return status.FAILURE

        if len(faces) > 1:
            logging.warn('Multiple faces detected in the frame')

        io.faces = faces

        return status.SUCCESS

class EmotionDetection(Stage):
    def __init__(self):
        super(EmotionDetection, self).__init__(input_keys=['faces', 'gray_frame', 'models'], output_keys=['predictions'])

    def execute(self, io, static_io):
        predictions = []
        
        for (x, y, w, h) in io.faces:
            face = io.gray_frame[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            try:
                predictions.append(static_io.models['emotion_detection_model'].predict(face[np.newaxis, :, :, np.newaxis]).tolist()[0])
            except Exception as e:
                logging.error(e)
                return status.FAILURE

        io.predictions = predictions

class StressClassifier(Stage):
    def __init__(self):
        super(StressClassifier, self).__init__(input_keys=['predictions', 'models'], output_keys=['stress_scores'])

    def execute(self, io, static_io):
        classifier_predictions = static_io.models['stress_classifier_model'].predict_proba(io.predictions)
        io.stress_scores = self.calculate_score(classifier_predictions, [0.5, 0.01, 0.95]).round(2)
        return status.SUCCESS

    def calculate_score(self, predict_probabilites, target_weights):
        scores = []
        normalising_value = np.array(target_weights).max()
        for prediction in predict_probabilites:
            score = 0
            for target_index in range(len(prediction)):
                score += (target_weights[target_index] * prediction[target_index]) / normalising_value
            scores.append(score * 100)
        
        return np.array(scores)
    
class GenerateDisplayFrame(Stage):
    def __init__(self):
        super(GenerateDisplayFrame, self).__init__(input_keys=['frame', 'faces', 'display_frame', 'stress_scores'], output_keys=['display_frame', 'display_frame_bytes'])

    def execute(self, io, static_io):
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_color = (255, 0, 0)
        thickness = 2

        face_index = 0
        io.display_frame = io.frame.copy()

        for (x, y, w, h) in io.faces:
            text = str(io.stress_scores[face_index]) + " " + "%"
            cv2.putText(io.display_frame, 
                text,
                (x,y),
                font_type, font_scale, font_color,
                thickness
                )
            
            center_x = int((x + x+w)/2)
            center_y = int((y + y+w)/2)
            radius = int(w/2)
            cv2.circle(io.display_frame,
                (center_x, center_y), radius,
                (0, 255, 0),
                thickness
                )

        _, jpeg = cv2.imencode('.jpg', io.display_frame)
        io.display_frame_bytes = jpeg.tobytes()
        
        return status.SUCCESS