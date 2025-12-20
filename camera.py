import os
import cv2
from PIL import Image
from numpy import asarray, expand_dims, linalg
from keras_facenet import FaceNet
import numpy as np

class VideoCamera(object):
    def __init__(self):
        # Initialize camera
        # Try camera index 1 first (external), then 0 (internal/default)
        self.video = cv2.VideoCapture(1)
        if not self.video.isOpened():
            print("⚠️  Camera 1 not found. Trying camera 0...")
            self.video = cv2.VideoCapture(0)
        
        if not self.video.isOpened():
            print("❌ CRITICAL ERROR: Camera access failed. Running in fallback mode (static image).")
            self.video = None

        # Initialize models
        self.haar_cascade = cv2.CascadeClassifier(cv2.samples.findFile(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
        self.facenet = FaceNet()
        
        # Load database
        self.database = {}
        self.load_database()

    def __del__(self):
        if self.video:
            self.video.release()

    def load_database(self):
        folder = 'images/'
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found. Database will be empty.")
            return

        print("Loading face database...")
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            try:
                gbr1 = cv2.imread(path)
                if gbr1 is None: continue
                
                wajah = self.haar_cascade.detectMultiScale(gbr1, 1.1, 4)
                
                if len(wajah) > 0:
                    x1, y1, width, height = wajah[0]
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1 + width, y1 + height
                    
                    gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
                    gbr = Image.fromarray(gbr)
                    gbr_array = asarray(gbr)
                    
                    face = gbr_array[y1:y2, x1:x2]
                    face = Image.fromarray(face)
                    face = face.resize((160, 160))
                    face = asarray(face)
                    
                    face = expand_dims(face, axis=0)
                    signature = self.facenet.embeddings(face)
                    
                    self.database[os.path.splitext(filename)[0]] = signature
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        print(f"Database loaded with {len(self.database)} identities.")

    def get_frame(self):
        if not self.video:
            # Return a blank or static frame if no camera
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "No Camera Found", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            ret, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()

        success, image = self.video.read()
        if not success:
            return None

        # Face Detection & Recognition Logic
        wajah = self.haar_cascade.detectMultiScale(image, 1.1, 4)

        if len(wajah) > 0:
            for (x1, y1, width, height) in wajah:
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height

                # Verify coordinates are within image bounds
                if x1 >= image.shape[1] or y1 >= image.shape[0]: continue
                
                # Extract face for recognition
                try:
                    face_img_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face_img_pil = Image.fromarray(face_img_cv)
                    face_array = asarray(face_img_pil)
                    
                    face_crop = face_array[y1:y2, x1:x2]
                    
                    if face_crop.size == 0: continue

                    face_pil = Image.fromarray(face_crop)
                    face_pil = face_pil.resize((160, 160))
                    face_np = asarray(face_pil)
                    face_input = expand_dims(face_np, axis=0)
                    
                    signature = self.facenet.embeddings(face_input)
                    
                    min_dist = 100
                    identity = 'Unknown'
                    
                    for key, value in self.database.items():
                        dist = linalg.norm(value - signature)
                        if dist < min_dist:
                            min_dist = dist
                            identity = key
                    
                    # Threshold for recognition (optional, keeping logic simple as per original)
                    if min_dist > 1.0: # Arbitrary threshold, can adjust
                         identity = 'Unknown' if min_dist > 0.8 else identity # Heuristic

                    # Draw box and label
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, identity, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.9, (255, 255, 0), 2, cv2.LINE_AA)
                except Exception as e:
                    print(f"Face processing error: {e}")
                    pass

        # Encode frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
