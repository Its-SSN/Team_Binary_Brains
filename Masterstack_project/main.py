from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from pygame import mixer

mixer.init()
sound = mixer.Sound('alarm.wav')

#model
class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, (3, 3))
    self.conv2 = nn.Conv2d(32, 32, (3, 3))
    self.conv3 = nn.Conv2d(32, 64, (3, 3))
    self.fc1 = nn.Linear(64*18*18, 128)
    self.fc2 = nn.Linear(128, 1)
  
  def forward(self, x):
    x = f.relu(self.conv1(x))  # n, 32, 22, 22
    x = f.relu(self.conv2(x))  # n, 32, 20, 20
    x = f.relu(self.conv3(x))  # n, 64, 18, 18

    x = x.view(-1, 64*18*18)

    x = f.dropout(x, 0.25)
    x = f.relu(self.fc1(x))
    x = f.dropout(x, 0.5)
    x = torch.sigmoid(self.fc2(x))
    return x

face_cascade = cv2.CascadeClassifier('haar_cascades/haarcascade_frontalface_alt2.xml')
l_eye_cascade = cv2.CascadeClassifier('haar_cascades\haarcascade_lefteye_2splits.xml')
r_eye_cascade = cv2.CascadeClassifier('haar_cascades\haarcascade_righteye_2splits.xml')

model = torch.load('model.pth')
  
app = Flask(__name__)
  
  
app.secret_key = 'xyzsdfg'
  
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'ssn10032004'
app.config['MYSQL_DB'] = 'login'
  
mysql = MySQL(app)
  
@app.route('/')
@app.route('/login', methods =['GET', 'POST'])
def login():
    mesage = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s AND password = % s', (email, password, ))
        user = cursor.fetchone()
        if user:
            session['loggedin'] = True
            session['username'] = user['username']
            session['password'] = user['password']
            session['email'] = user['email']
            mesage = 'Logged in successfully !'
            return render_template('user.html', mesage = mesage)
        else:
            mesage = 'Please enter correct email / password !'
    return render_template('login.html', mesage = mesage)
  
@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('username', None)
    session.pop('email', None)
    return redirect(url_for('login'))
  
@app.route('/register', methods =['GET', 'POST'])
def register():
    mesage = ''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form :
        userName = request.form['name']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s', (email, ))
        account = cursor.fetchone()
        if account:
            mesage = 'Account already exists !'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            mesage = 'Invalid email address !'
        elif not userName or not password or not email:
            mesage = 'Please fill out the form !'
        else:
            cursor.execute('INSERT INTO user VALUES ( % s, % s, % s)', (userName, email, password, ))
            mysql.connection.commit()
            mesage = 'You have successfully registered !'
    elif request.method == 'POST':
        mesage = 'Please fill out the form !'
    return render_template('register.html', mesage = mesage)


@app.route('/webcam')
def webcam():
    cap = cv2.VideoCapture(0)
    score = 0

    while True:
        l_pred = 0
        r_pred = 0
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        l_eye = l_eye_cascade.detectMultiScale(gray, 1.3, 15)
        r_eye = r_eye_cascade.detectMultiScale(gray, 1.3, 15)
        for (x, y, h, w) in faces:
            cv2.rectangle(frame, (x, y), (x+h, y+w), (255, 0, 0), 2)
            cv2.putText(frame, "FACE", (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 0, 0), 1)

        for (x, y, h, w) in l_eye:
            cv2.rectangle(frame, (x, y), (x+h, y+w), (0, 255, 0), 2)
            cv2.putText(frame, "L EYE", (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

            l_eye_img=frame[y:y+h,x:x+w]
            l_eye_img = cv2.cvtColor(l_eye_img,cv2.COLOR_BGR2GRAY)
            l_eye_img = cv2.resize(l_eye_img,(24,24))
            l_eye_img = torch.tensor(l_eye_img.reshape(-1, 24, 24), dtype=torch.float)
            l_pred = model(l_eye_img).item()

        for (x, y, h, w) in r_eye:
            cv2.putText(frame, "R EYE", (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (x, y), (x+h, y+w), (0, 255, 0), 2)

            r_eye_img=frame[y:y+h,x:x+w]
            r_eye_img = cv2.cvtColor(r_eye_img,cv2.COLOR_BGR2GRAY)
            r_eye_img = cv2.resize(r_eye_img,(24,24))
            r_eye_img = torch.tensor(r_eye_img.reshape(-1, 24, 24), dtype=torch.float)
            r_pred = model(r_eye_img).item()
                
        if l_pred > 0.9 and r_pred > 0.9:
            score += 1
        else:
            score -= 3
            score = max(0, score)
        
        height, width = frame.shape[:2]
        cv2.putText(frame, str(score), (20, height-20), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

        if score > 20:
            try:
                sound.play()
            except:
                pass
        elif score < 5:
            try:
                sound.stop()
            except:
                pass

        cv2.imshow('feed', frame)
        k = cv2.waitKey(1) & 0xff
        if k== 27:
            break

    cap.release()
    cv2.destroyAllWindows()   
    return render_template('user.html')
    
if __name__ == "__main__":
    app.run(debug=True)