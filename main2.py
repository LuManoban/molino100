from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
import cv2
from ultralytics import YOLO
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import numpy as np

class MolinoApp(App):

    def build(self):
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.capture.set(3, 640)
        self.capture.set(4, 480)
        self.model = YOLO('Modelos/molinosag.pt')
        self.clsName = ['MOLINOSAG', 'PERNOS', 'LINERS']
        Clock.schedule_interval(self.update, 1.0 / 30.0) # 30fps
        self.root = Builder.load_file("molino.kv")
        return self.root

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame_processed = self.process_frame(frame)
            self.root.ids.video_display.texture = self.get_texture_from_frame(frame_processed)

            # Diccionario para almacenar las detecciones
            detections = {}

            results = self.model(frame, stream=True, verbose=False)
            for res in results:
                boxes = res.boxes
                for box in boxes:
                    # Boonding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Error
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0
                    if x2 < 0: x2 = 0
                    if y2 < 0: y2 = 0

                    # Clase
                    cls = int(box.cls[0])

                    # Confidence
                    conf = box.conf[0]
                    CONFIDENCE_THRESHOLD = 0.90
                    print(f"Clase: {cls}, Confianza: {conf}")
                    if conf > CONFIDENCE_THRESHOLD:
                        # Almacena la detección
                        detections[cls] = conf

            # Actualiza los widgets de imagen en función de las detecciones
            paths = {0: 'setUp/infgeneral.png', 1: 'setUp/pernos.png', 2: 'setUp/liners.png'}
            widget_ids = {0: 'detected_class_display_1', 1: 'detected_class_display_2', 2: 'detected_class_display_3'}

            for cls, widget_id in widget_ids.items():
                if cls in detections:
                    self.root.ids[widget_id].source = paths[cls]
                else:
                    self.root.ids[widget_id].source = ''

    def process_frame(self, frame):
            frame_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(frame, stream=True, verbose=False)
            for res in results:
                boxes = res.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Error handling
                    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)

                    # Class and Confidence
                    cls = int(box.cls[0])
                    conf = box.conf[0]
                    CONFIDENCE_THRESHOLD = 0.90

                    if conf > CONFIDENCE_THRESHOLD:
                        color = None
                        if cls == 0:
                            color = (255, 0, 0)
                        elif cls == 1:
                            color = (255, 128, 0)
                        elif cls == 2:
                            color = (0, 255, 0)

                        # Draw rectangle
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), color, 2)

                        # Draw text
                        text = f'{self.clsName[cls]} {int(conf * 100)}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0, 0, 0),
                                      cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            return frame_show

    def get_texture_from_frame(self, frame):
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')  # Cambia 'bgr' a 'rgb'
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')  # Cambia 'bgr' a 'rgb'
        return texture



if __name__ == '__main__':
    MolinoApp().run()