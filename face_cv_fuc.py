import cv2,os
import numpy as np
import time
import serial
from PIL import Image
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

def thermal():
    global dataarr,hexi
    hexi = ''
    dataarr = []
    arr = 0
    ser.open()
    ser.write(START)
    buf = ser.read(39974)
    ser.write(STOP)
    ser.close()
    bufsp = buf.split(b'\x00\x00\r\n') #Cut stopbit
    bufi = bufsp[0][9:]

    for i in (number+1 for number in range(len(bufsp)-3)): #0-119 line
        bufi += bufsp[i][9:] #Cut startbit
        lens = len(bufsp)-3

    if(len(bufi) == 38400):
        image1 = Image.frombytes('I;16', (160,120), bufi, 'raw')
        #image1.show(title="L")
        arr = np.array(image1)

    hexi = bufi.hex()

    intarr = [int(hexi[k:k+4],16) for k in range(0,len(hexi),4)]
    for n in range(len(intarr)):
      intarr[n] = (intarr[n]/10)-273
      dataarr.append(intarr)

    return arr,dataarr

ser = serial.Serial('/dev/ttyUSB0',
    baudrate=8000000,
    bytesize=8, 
    parity='N', 
    stopbits=1,
    timeout=2,
    writeTimeout=2)

ser.close()

START = [0xAA, 0x55, 0x00, 0x00, 0x00, 0x01, 0x4F, 0x7E, 0x00, 0x00, 0x00, 0x0D, 0x0A]
STOP = [0xAA, 0x55, 0x00, 0x00, 0x00, 0x01, 0x4F, 0x7E, 0xFF, 0x1E, 0xF0, 0x0D, 0x0A]
AT_shutter = [0xAA, 0x55, 0x00, 0x04, 0x00, 0x01, 0x93, 0xBE, 0x00, 0x00, 0x00, 0x0D, 0x0A]
S_shutter = [0xAA, 0x55, 0x00, 0x04, 0x00, 0x01, 0x93, 0xBE, 0xFF, 0x1E, 0xF0, 0x0D, 0x0A]
TEMP = [0xAA, 0x55, 0x00, 0x09, 0x00, 0x01, 0xD1, 0xEF, 0x00, 0x00, 0x00, 0x0D, 0x0A]
GREY = [0xAA, 0x55, 0x00, 0x09, 0x00, 0x01, 0xD1, 0xEF, 0xFF, 0x1E, 0xF0, 0x0D, 0x0A]
TEMP2 = [0xAA, 0x55, 0x00, 0x09, 0x00, 0x01, 0xD1, 0xEF, 0x00, 0x00, 0x00, 0x0D, 0x0A]
#Set center 35c 
SET = [0xAA, 0x55, 0x00, 0x21, 0x00, 0x04, 0xAE, 0x2D, 0x00, 0x00, 0x0C, 0x42, 0x2D, 0xEB, 0x0D, 0x0A]

ser.open()
ser.write(AT_shutter)
ser.close()
time.sleep(0.5)
ser.open()
ser.write(TEMP)
ser.close()
print ("Set center 35C......")
iset = 4
while iset > 0:
    iset -= 1
    print (iset)
    time.sleep(1)
ser.open()
ser.write(SET)
ser.close()
print ("Done..")
time.sleep(1)
arr,data = thermal()
print ("Center point is ",str(data[60][80]))
time.sleep(1)

count_ = 0
#cam = cv2.VideoCapture('v4l2src device=/dev/video0 ! video/x-raw,framerate=30/1 ! videoscale ! video/x-raw,width=320,height=260 ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
cam = cv2.VideoCapture(0)
cam.set(3, 375)
cam.set(4, 290)
while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_ = frame[30:30+220, 20:20+300]
    start_time = time.time()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    gray = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.05, 5)
    arr,data = thermal()
    arr=arr/255
    
    temp_center = float("{:.1f}".format(data[60][80]))
    cv2.rectangle(arr, (0, 0), (0 + 80, 0 + 32), (128,128,128), -1)
    cv2.putText(arr, str("Center ="+str(temp_center)), (1, 10), font, 0.5, (0, 0, 0), 1)
    for (x, y, w, h) in faces:
        x_ = (x//2)+20
        y_ = (y//2)-10
        w_ = w//2
        h_ = h//2
        xcenter = ((x_ + x_+w_) // 2)
        ycenter = ((y_ + y_+h_) // 2)-20
        cv2.circle(arr, (xcenter,ycenter), 2, (0,0,0), -1)

        if(h > 80 ):
            temp_face = float("{:.2f}".format(data[ycenter][xcenter]))
            cv2.rectangle(arr, (x_, y_), (x_+w_, y_+h_), (255, 0, 0), 1)
            cv2.rectangle(frame_, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv2.putText(arr, str(temp_face), (xcenter,ycenter), font, 0.4, (255, 255, 255),1)
            cv2.putText(arr, str("Face ="+str(temp_face)), (1, 30), font, 0.5, (0, 0, 0), 1)
            if(count_ > 10):
                count_ = 0
                pause = 1
                face_img = frame_[y:y+h,x:x+w]
                if(temp_face < 35):
                    cv2.putText(frame, "Your temp : "+str(temp_face), (10,280), font, 1, (10, 255, 10),2)
                else:
                    cv2.putText(frame, "Your temp : "+str(temp_face), (10,280), font, 1, (10, 10, 255),2)
                cv2.imshow('face', frame)
    count_ += 1
    arr = cv2.resize(arr,(320,280))
    frame_ = cv2.resize(frame_,(320,280))
    try:
        cv2.imshow('camera_crop', frame_)
        cv2.imshow('Heat16', arr.astype('uint8'))
    except:
        print('shutter')
    # wait for 100 miliseconds
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
        # break if the sample number is morethan 100
cam.release()
cv2.destroyAllWindows()

