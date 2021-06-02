# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:57:44 2019

@author: seraj
"""
import numpy as np
import imutils
import sys
import cv2
from flask import Flask, request, render_template,Response
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    """Video streaming home page."""
    if request.method=="POST":
        path = request.form.get("path")
        return render_template('result.html',path1=path)
    else:
        return render_template('index.html')
       
def gen(path):
    CLASSES = open("action_recognition_kinetics.txt").read().strip().split("\n")
    SAMPLE_DURATION = 16
    SAMPLE_SIZE = 112
    print(path)
    # load the human activity recognition model
    print("[INFO] loading human activity recognition model...")
    net = cv2.dnn.readNet("resnet-34_kinetics.onnx")

    # grab a pointer to the input video stream
    print("[INFO] accessing video stream...")
    vs = cv2.VideoCapture(path if path else 0)
    c=0
    # loop until we explicitly break from it
    h=1
    all=[]
    count=0
    while True:
        frames = []
        c=c+1
        l=0
        # loop over the number of required sample frames
        for i in range(0, SAMPLE_DURATION):
            # read a frame from the video stream
            (grabbed, frame) = vs.read()
            l=l+1
            # if the frame was not grabbed then we've reached the end of
            # the video stream so exit the script
            if not grabbed:
                print("[INFO] no frame read from stream - exiting")
                h=0
                break

            # otherwise, the frame was read so resize it and add it to
            # our frames list
            frame = imutils.resize(frame, width=400)
            frames.append(frame)
        if h==0:
            break
        # now that our frames array is filled we can construct our blob
        blob = cv2.dnn.blobFromImages(frames, 1.0,
                                        (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
                                        swapRB=True, crop=True)
        blob = np.transpose(blob, (1, 0, 2, 3))
        blob = np.expand_dims(blob, axis=0)

        # pass the blob through the network to obtain our human activity
        # recognition predictions
        net.setInput(blob)
        outputs = net.forward()
        label = CLASSES[np.argmax(outputs)]
        for frame in frames:
            # draw the predicted activity on the frame
            framenext=cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)       
            img = cv2.resize(framenext, (0,0), fx=1.5, fy=1.5) 
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/video_feed/<string:path>')
def video_feed(path):
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



    

