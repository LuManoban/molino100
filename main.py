#Libraries
from tkinter import *
from PIL import Image, ImageTk
import imutils
import cv2
import numpy as np
from ultralytics import YOLO
import math


def render_Image (type):


    if type == 0:
        img = cv2.imread('setUp/infgeneral.png')
        lblimg.place(x=75, y=260)

    elif type == 1:
        img = cv2.imread('setUp/pernos.png')
        lblimg.place(x=995,y=150)

    elif type == 2:
        img = cv2.imread('setUp/liners.png')
        lblimg.place(x=995, y=400)

    else:
        img4 = cv2.imread('setUp/tabla.png')

    img = np.array(img, dtype='uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    img_ = ImageTk.PhotoImage(image=img)
    lblimg.configure(image=img_)
    lblimg.image = img_




#Scanning Function
def Scanning():

    #Read videocapture
    if cap is not None:
        ret, frame = cap.read()
        frame_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret == True:

            results = model(frame, stream=True, verbose=False)
            for res in results:
                # Box
                boxes = res.boxes
                for box in boxes:
                    # Boonding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    #Error
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0
                    if x2 < 0: x2 = 0
                    if y2 < 0: y2 = 0

                    #Clase
                    cls= int(box.cls[0])

                    #Confidence
                    conf = box.conf[0]
                    CONFIDENCE_THRESHOLD = 0.90
                    print(f"Clase: {cls}, Confianza: {conf}")

                    if conf > CONFIDENCE_THRESHOLD:
                        if cls == 0:
                            #Draw Rectangulo
                            cv2.rectangle(frame_show, (x1,y1), (x2,y2), (255,255,0),2)

                            #text
                            text = f'{clsName[cls]} {int(conf * 100)}%'
                            sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,1 ,2)
                            dim = sizetext[0]
                            baseline = sizetext[1]
                            #Rect
                            cv2.rectangle(frame_show, (x1,y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0,0,0), cv2.FILLED)
                            cv2.putText(frame_show, text, (x1,y1 -5), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0),2)
                            render_Image(cls)


                        elif cls == 1:
                            # Draw Rectangulo
                            cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 255, 0), 2)

                            # text
                            text = f'{clsName[cls]} {int(conf * 100)}%'
                            sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                            dim = sizetext[0]
                            baseline = sizetext[1]
                            # Rect
                            cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0, 0, 0),
                                          cv2.FILLED)
                            cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 0), 2)
                            render_Image(cls)

                        elif cls == 2:
                            # Draw Rectangulo
                            cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 255, 0), 2)

                            # text
                            text = f'{clsName[cls]} {int(conf * 100)}%'
                            sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                            dim = sizetext[0]
                            baseline = sizetext[1]
                            # Rect
                            cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0, 0, 0),
                                          cv2.FILLED)
                            cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            render_Image(cls)
                        #else:
                            #lblimg.config(image='')


            #Resize
            frame_show = imutils.resize(frame_show, width=640)

            #Convwetir Video
            im = Image.fromarray(frame_show)
            img = ImageTk.PhotoImage(image=im)

            #Mostrar
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, Scanning)
        else:
            cap.release()

#main
def ventana_principal():
    global model , clsName, img_generaltxt, img_linerstxt, img_pernostxt, cap, lblVideo, pantalla, lblimg
    # Ventana principal
    pantalla = Tk()
    pantalla.title("MOLINO SAG")
    pantalla.geometry("1281x721")
    lblimg = Label(pantalla)
    #background
    imagenF = PhotoImage(file="setUp/Ventanaprinc.png")
    background = Label(image=imagenF)
    background.place(x=0, y=0, relwidth=1, relheight=1)

    #Model
    model = YOLO('Modelos/molinosag.pt')

    #Clases
    clsName = ['MOLINOSAG', 'PERNOS' , 'LINERS']

    #Img
    #img_generaltxt = cv2.imread('setUp/general.png')
    #img_linerstxt = cv2.imread('setUp/liners.PNG')
    #img_pernostxt = cv2.imread('setUp/pernos.png')

    # Label Video
    lblVideo = Label(pantalla)
    lblVideo.place(x=330, y=150)

    #Cam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 640)
    cap.set(4,480)

    #Scanning
    Scanning()

    #Cam
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(3, 640)
    cap.set(4,480)


    #Loop
    pantalla.mainloop()

if __name__ == '__main__':
    ventana_principal()


