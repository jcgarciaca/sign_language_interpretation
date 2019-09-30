import tkinter
from tkinter import filedialog
import cv2
import PIL.Image, PIL.ImageTk
import time
import os
from network_model import NetworkModel

class App:
    def __init__(self, window, window_title, video_source=0):
        self.nn_model = NetworkModel()

        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        
        self.init_complete = False
        
        self.text_disp = tkinter.StringVar()
        self.text_disp.set('Meaning')

        # open video source (by default this will try to open the computer webcam)
        self.vid = None

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = 600, height = 400)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_open = tkinter.Button(window, text="Open Video", width=20, command=self.search_file)
        self.btn_analyze = tkinter.Button(window, text="Analyze", width=20, command=self.analyze_file)
        self.lbl_pred = tkinter.Label(window, textvariable=self.text_disp, font=("Helvetica", 16))
        self.btn_open.pack(anchor=tkinter.SW, expand=True)
        self.btn_analyze.pack(anchor=tkinter.SW, expand=True)
        self.lbl_pred.pack(anchor=tkinter.S, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()
        
    def search_file(self):
        self.filename = filedialog.askopenfilename(initialdir = '/home/msdc/Downloads/señas', title = "Select video")
        self.vid = MyVideoCapture(self.filename)
        if not self.init_complete:
            self.init_complete = True

    def analyze_file(self):
        prediction = self.nn_model.predict_video(self.filename)
        self.text_disp.set(prediction)#'Prediction {}'.format(self.cnt))
            
    def update(self):
        # Get a frame from the video source
        if self.init_complete:
            ret, frame = self.vid.get_frame()

            if ret:
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
                
        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                frame = resized = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_AREA)
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
App(tkinter.Tk(), "Sign language prediction")
