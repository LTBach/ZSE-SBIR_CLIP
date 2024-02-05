from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.colorchooser import askcolor
from PIL import Image, ImageDraw, ImageTk

import os
import argparse
import json as js
import cv2 as cv

import torch

from utils.util import get_output_file, load_checkpoint
from options import Option
from model.model import Model
from retrieval import retrieve


class Paint(Frame):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = "black"
    WIDTH = 672
    HEIGHT = 672

    def __init__(self, master, sketch_dir="sketchs"):

        super().__init__(master)

        self.pen_button = Button(self, text="pen", command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.erase_button = Button(self, text="eraser", command=self.use_eraser)
        self.erase_button.grid(row=0, column=1)

        self.choose_cursor_size_slider = Scale(self, orient=HORIZONTAL)
        self.choose_cursor_size_slider.grid(row=0, column=2)

        self.save_button = Button(self, text="save", command=self.save, relief=GROOVE)
        self.save_button.grid(row=0, column=3)

        self.clear_button = Button(self, text="clear", command=self.clear, relief=GROOVE)
        self.clear_button.grid(row=0, column=4)

        self.load_button = Button(self, text="load", command=self.load, relief=GROOVE)
        self.load_button.grid(row=0, column=5)

        self.canva = Canvas(self, bg="white", width=self.WIDTH, height=self.HEIGHT)
        self.canva.grid(row=1, column=0, columnspan=6)

        self.setup()

        self.image = Image.new("RGB", (self.WIDTH, self.HEIGHT), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # Path
        self.current_path = os.getcwd()
        self.sketch_path = os.path.join(self.current_path, sketch_dir)

        os.makedirs(self.sketch_path, exist_ok=True)

    def setup(self):

        # Default
        self.old_x = None
        self.old_y = None
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.activate_button(self.pen_button)

        self.config_scale(1, 100, 5)

        self.canva.bind("<B1-Motion>", self.paint)
        self.canva.bind("<ButtonRelease-1>", self.reset)
        self.canva.config(cursor = "pencil")

    def use_pen(self):
        self.activate_button(self.pen_button)
        self.config_scale(1, 100, 5)
        self.canva.config(cursor = "pencil")

    def use_eraser(self):
        self.activate_button(self.erase_button, eraser_mode=True)
        self.config_scale(1, 100, 40)
        self.canva.config(cursor = "draped_box")

    def save(self):
        sketch_path, sketch_id = get_output_file(self.sketch_path)
        self.image.save(sketch_path)
        return sketch_path, sketch_id

    def clear(self):
        self.canva.delete("all")
        self.image = Image.new("RGB", (self.WIDTH, self.HEIGHT), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def load(self):

        filetypes = (
            ('png images', '*.png'),
            ('jpg images', '*.jpg'),
            ('All files', '*.*')
        )

        # file_path = fd.askopenfilename()
        file_path = fd.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes)

        if file_path[-3:] != "png":
            im = Image.open(file_path)
            im_name = file_path.split('\\')[-1]
            new_path = os.path.join('tmp', im_name[:-3] + 'png')
            im.save(new_path)
            file_path = new_path

        self.image = Image.open(file_path).resize((self.WIDTH, self.HEIGHT))
        self.draw = ImageDraw.Draw(self.image)

        self.ph = ImageTk.PhotoImage(self.image)
        self.canva.create_image((self.WIDTH//2, self.HEIGHT//2), image=self.ph)
        #self.canva.itemconfig(self.canva_image)


    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.cursor_size = self.choose_cursor_size_slider.get()

        paint_color = "white" if self.eraser_on else self.color
        
        if self.old_x and self.old_y:
            ox, oy, ex, ey = self.old_x, self.old_y, event.x, event.y
            self.canva.create_line(ox, oy, ex, ey,
                               width=self.cursor_size, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            if paint_color != "white":
                self.draw.line([(ox, oy), (ex, ey)], width=self.cursor_size, fill = (0,0,0))
            else:
                self.draw.line([(ox, oy), (ex, ey)], width=self.cursor_size + 5)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def config_scale(self, from_ = 1, to = 10, default=2):
        self.choose_cursor_size_slider.config(from_=from_)
        self.choose_cursor_size_slider.config(to=to)
        self.choose_cursor_size_slider.set(default)

class ImageField(Frame):
    def __init__(self, master):
        super().__init__(master)

        self.choose_num_img_slider = Scale(self, orient=HORIZONTAL)
        self.choose_num_img_slider.grid(row=0, columnspan=4)

        self.setup()

    def setup(self):
        self.config_scale(1, 16, 8)

    def show_image(self, path_list):

        not_jpg_path = []
        for path in path_list:
            if path[-3:] != "png":
                im = Image.open(path)
                im_name = path.split('\\')[-1]
                new_path = os.path.join('tmp', im_name[:-3] + 'png')
                im.save(new_path)
                not_jpg_path.append(new_path)
            else:
                not_jpg_path.append(path)


        self.images = []
        self.img_buttons = []

        i = 0

        for path in not_jpg_path:
            self.photo = PhotoImage(file=path) #didn"t support jpg :)))
            self.button = Button(self, height = 224, width = 224)
            self.img_buttons.append(self.button)
            self.images.append(self.photo)
            self.img_buttons[i].config(image = self.images[i])
            self.img_buttons[i].grid(row = i//4 + 1, column = i%4)
            
            i+=1

    def config_scale(self, from_ = 1, to = 10, default=2):
        self.choose_num_img_slider.config(from_=from_)
        self.choose_num_img_slider.config(to=to)
        self.choose_num_img_slider.set(default)

class App(Tk):
    def __init__(self, args):
        super().__init__()
        self.geometry("2560x1600+0+0")
        self.title("Sketch Based Image Retrieval demo")

        self.DrawField = Paint(self)
        self.DrawField.grid(row = 0, column = 0)

        self.retrieve_button = Button(self, text='Search', command=self.Retrieve)
        self.retrieve_button.grid(row = 1, column = 0)

        self.ImageField = ImageField(self)
        self.ImageField.grid(row = 0, column = 1)

        # Path
        self.current_path = os.getcwd()
        self.result_path = os.path.join(self.current_path, args.output_dir)

        os.makedirs(self.result_path, exist_ok=True)

        self.args = args
        self.setup()

    def setup(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Model(args)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.model.to(self.device)
        
        if os.path.isfile(args.load):
            checkpoint = load_checkpoint(args.load)
            cur = self.model.state_dict()
            new = {k: v for k, v in checkpoint['model'].items() if k in cur.keys()}
            cur.update(new)
            self.model.load_state_dict(cur)  
        else:
            raise ImportError("Pre-trained weigths not found!")


    def Retrieve(self):
        num_img = self.ImageField.choose_num_img_slider.get()

        sketch_path, sketch_id = self.DrawField.save()
        result_path = os.path.join(self.result_path, f"result-{sketch_id}")
        path_list = retrieve(self.device, sketch_path, "Sketchy", self.model, False, args, result_path, k=num_img)
        self.ImageField.show_image(path_list)

if __name__ == "__main__":
    args = Option().parse()

    app = App(args)
    app.mainloop()
