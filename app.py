from flask import Flask, render_template, Response
from camera import VideoCamera
import os

# Suppress OpenCV logs
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

app = Flask(__name__)

# Initialize camera on startup or first request
camera = None

def get_camera():
    global camera
    if camera is None:
        camera = VideoCamera()
    return camera

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            break

@app.route('/video_feed')
def video_feed():
    return Response(gen(get_camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Host 0.0.0.0 is important for Docker
    # app.run(host='0.0.0.0', port=5000, debug=False)
    app.run(host='localhost', port=1340, debug=False)
