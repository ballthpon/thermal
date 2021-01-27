from flask import Flask, render_template, Response
import cv2
import os
import numpy as np
import tensorflow as tf
import sys
import requests
import time
from line_notify import LineNotify


sys.path.append("..")
CWD_PATH = os.getcwd()

url = 'https://notify-api.line.me/api/notify'
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
token = "O63M1FDJtY6dwM17aUS5oC5uWe1SG0gAdRObGTa9uO4"
headers = {
            'content-type':
            'application/x-www-form-urlencoded',
            'Authorization':'Bearer '+token
           }

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,'mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

labels = []
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

IM_WIDTH = 640
IM_HEIGHT = 480
cap = cv2.VideoCapture(0)
cap.set(3,IM_WIDTH)
cap.set(4,IM_HEIGHT)

font = cv2.FONT_HERSHEY_SIMPLEX
c_f = 0
score_to_web = 0
class_to_web = 0

xxmin = 0
xxmax = 0
yymin = 0
yymax = 0 

app = Flask(__name__)

@app.route('/')

def index():
    """Video streaming home page."""
    
    return render_template('index.html')

def pet_detector(frame):

    global yymin,xxmin,yymax,xxmax,score_to_web,class_to_web
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    si = 0.0
    ii = 0
    for i in range(100):
        min_conf_threshold = 0.8
        if (scores[0][i] >= min_conf_threshold):
            if (scores[0][i] > si):
                si = scores[0][i]
                ii = i
                score_to_web = "{:.2f}".format(si*100)
                class_to_web = str(labels[int(classes[0][i])])
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
    ymin = int((boxes[0][ii][0]*IM_HEIGHT))
    xmin = int((boxes[0][ii][1]*IM_WIDTH))
    ymax = int((boxes[0][ii][2]*IM_HEIGHT))
    xmax = int((boxes[0][ii][3]*IM_WIDTH))

    yymin = ymin
    xxmin = xmin
    yymax = ymax
    xxmax = xmax
    # Check the class of the top detected object by looking at classes[0][0].
    # If the top detected object is a cat (17) or a dog (18) (or a teddy bear (88) for test purposes),
    # find its center coordinates by looking at the boxes[0][0] variable.
    # boxes[0][0] variable holds coordinates of detected objects as (ymin, xmin, ymax, xmax)
    Result = np.array(frame[ymin-50:ymax+50,xmin-50:xmax+50])
    cv2.imwrite("notify3.jpg", Result)
    time.sleep(0.3) 
    return frame

def gen():
    """Video streaming generator function."""
    global c_f
    while True:
        ret,frame = cap.read()
        if c_f > 6:
            frame = pet_detector(frame)
            c_f = 0
        c_f += 1
        
        cv2.rectangle(frame, (xxmin,yymin), (xxmax,yymax), (0, 0, 255), 2)
        time.sleep(0.2)
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

def cap_gen():
    while True:
        time.sleep(0.7)
        cap_img = cv2.imread("notify3.jpg")
        try:
            (flag, encodImage) = cv2.imencode(".jpg", cap_img)
            yield (b'--cap\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodImage) + b'\r\n')
        except:
            print ("Noimg")
            #yield (b'--cap\r\n'
            #       b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_cap')
def video_cap():
    return Response(cap_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=cap')
@app.route('/get_class_score', methods=['GET'])
def get_class_score():
    data = str(class_to_web) + " :  " +str(score_to_web) + " %"
    print ("data : ",data)
    return data
@app.route('/get_time', methods=['GET'])
def get_time():
    t = time.localtime()
    data = time.strftime("%H:%M:%S", t)
    return data



if __name__ == '__main__':
    app.run(host='0.0.0.0', port =8000, debug=True, threaded=True, use_reloader=False)
