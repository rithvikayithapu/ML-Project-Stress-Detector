# Let us import the Libraries required.
import os
import cv2
import urllib
import numpy as np
from werkzeug.utils import secure_filename
from urllib.request import Request, urlopen
from flask import Flask, render_template, Response, request, redirect, flash, url_for

# Importing the required Classes/Functions from Modules defined.
from camera import VideoCamera
# Let us Instantiate the app
app = Flask(__name__)

###################################################################################
# We define some global parameters so that its easier for us to tweak when required.

# When serving files, we set the cache control max age to zero number of seconds
# for refreshing the Cache
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

###################################################################################
# Some Utility Functions

# Flask provides native support for streaming responses through the use of generator
# functions. A generator is a special function that can be interrupted and resumed.


def gen(camera):
    "" "Helps in Passing frames from Web Camera to server"""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def allowed_file(filename):
    """ Checks the file format when file is uploaded"""
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


###################################################################################

def mood(result):
    if result=="Happy":
        return 'Since you are happy, lets keep up the good mood with some amazing music!'
    elif result=="Sad":
        return 'It seems that you are having a bad day, lets cheer you up with some amazing music!'
    elif result=="Disgust":
        return 'It seems something has got you feeling disgusted. Lets improve your mood with some great music!'
    elif result=="Neutral":
         return 'It seems like a normal day. Lets turn it into a great one with some amazing music!'
    elif result=="Fear":
        return 'You seem very scared. We are sure that some music will help!'
    elif result=="Angry":
        return 'You seem angry. Listening to some music will surely help you calm down!'
    elif result=="Surprise":
        return 'You seem surprised! Hopefully its some good news. Lets celebrate it with some great music!'


def provide_url(result):
    if result=="Happy":
        return 'https://open.spotify.com/playlist/1BVPSd4dynzdlIWehjvkPj'
    elif result=="Sad":
        return 'https://www.writediary.com/ '
    elif result=="Disgust":
        return 'https://open.spotify.com'
    elif result=="Neutral":
         return 'https://www.netflix.com/'
    elif result=="Fear":
        return 'https://www.youtube.com/watch?v=KWt2-lUpg-E'
    elif result=="Angry":
        return 'https://www.onlinemeditation.org/'
    elif result=="Surprise":
        return 'https://www.google.com/search?q=hotels+near+me&oq=hotels+&aqs=chrome.1.69i57j0i433i457j0i402l2j0i433l4j0l2.3606j0j7&sourceid=chrome&ie=UTF-8'


def activities(result):
    if result == "Happy":
        return '• Try out some dance moves'


    elif result == "Sad":
        return '• Write in a journal'

    elif result == "Disgust":
        return '• Listen soothing music'

    elif result == "Neutral":
        return '• Watch your favourite movie'

    elif result == "Fear":
        return '• Get a good sleep'

    elif result == "Angry":
        return '• Do meditation'


    elif result == "Surprise":
        return '• Give yourself a treat' \


@app.route('/')
def Start():
    """ Renders the Home Page """

    return render_template('Start.html')


@app.route('/video_feed')
def video_feed():
    """ A route that returns a streamed response needs to return a Response object
    that is initialized with the generator function."""

    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/RealTime', methods=['POST'])
def RealTime():
    """ Video streaming (Real Time Image from WebCam Video) home page."""

    return render_template('RealTime.html')

if __name__ == '__main__':
    app.run(debug=True)
