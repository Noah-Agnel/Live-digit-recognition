from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
import pickle


class Paint(object):

    DEFAULT_PEN_SIZE = 40

    DEFAULT_COLOR = 'white'

    def __init__(self):
        self.root = Tk()


        self.erase_button = Button(self.root, text='erase all', command=self.erase)
        self.erase_button.grid(row=0, column=0)

        self.save_button = Button(self.root, text='save', command=self.save)
        self.save_button.grid(row=0, column=1)

        self.predict_button = Button(self.root, text='predict', command=self.predict)
        self.predict_button.grid(row=0, column=2)

        self.result_label = Label(text="Draw a digit")
        self.result_label.grid(row=1, column=1)

        self.c = Canvas(self.root, bg='black', width=20*28, height=20*28)
        self.c.grid(row=2, columnspan=3)

        self.setup()
        self.root.mainloop()
        self.network = None
    
    def setup(self):
        try:
            file = open("trainednet.pkl", 'rb')
            self.network = pickle.load(file)
        except Exception as e:
            return

        self.old_x = None
        self.old_y = None
        self.line_width = self.DEFAULT_PEN_SIZE
        self.color = self.DEFAULT_COLOR
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.image = Image.new("RGB", (20*28, 20*28), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)

    def erase(self):
        self.c.delete("all")
        self.draw.rectangle((0, 0, 20*28, 20*28), (0, 0, 0))

    def paint(self, event):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=self.color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.draw.line([self.old_x, self.old_y, event.x, event.y], width=self.line_width, fill=self.color)
        self.old_x = event.x
        self.old_y = event.y


    def reset(self, event):
        self.old_x, self.old_y = None, None

    def save(self):
        out_image = Image.new("RGB", (28, 28), (0, 0, 0))
        for i in range(28):
            for j in range(28):
                p_b = self.average_brightness(i,j)
                out_image.putpixel((i, j), (p_b,p_b,p_b))
        filename = "image.png"
        out_image.save(filename)

    def predict(self):
        out = np.array([[self.average_brightness(i,j)] for j in range(28) for i in range(28)])
        prediction = self.network.recognize(out/255)
        self.result_label.config(text="Predicted digit: " + str(prediction))

    def average_brightness(self, x, y):
        # function that gets the average colour of pixels on the 20x20 grid 
        # and returns the average brightness of the 28x28 grid
        total = 0
        for i in range(20):
            for j in range(20):
                total += self.image.getpixel((x*20+i, y*20+j))[0]

        return total//400

Paint()