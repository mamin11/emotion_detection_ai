import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2

def load_img():
    global img, img_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    img_data = filedialog.askopenfilename(initialdir="/", title="Choose Image", filetypes=(("all files", "*.*"), ("png files", "*.png")))
    basewidth = 250
    img = Image.open(img_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1] * float(wpercent))))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = img_data.split('/')
    panel = tk.Label(frame, text=str(file_name[len(file_name) -1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()

def classify_img():
    class_labels=['Angry','Disgusted','Afraid','Happy','Neutral','Sad','Surprised']

    original = Image.open(img_data).convert('L')
    original = original.resize((48,48), Image.ANTIALIAS)

    # original = original.astype('float')/255.0

    numpy_img = img_to_array(original)
    image_batch = np.expand_dims(numpy_img, axis=0)
    image_batch /= 255.

    # processed_image = classifier.prepocess_input(image_batch.copy())
    prediction = classifier.predict(image_batch)[0]
    label=class_labels[prediction.argmax()]
    # print(numpy_img)

    result = tk.Label(frame, text=label).pack()

root = tk.Tk()
root.title('EMOTION CLASSIFIER AI')
# root.iconbitmap('class.ico')
root.resizable(False, False)
tit = tk.Label(root, text="Emotion Classifier App", padx=25, pady=6, font=("", 12)).pack()

canvas = tk.Canvas(root, height=500, width=500, bg='grey')
canvas.pack()

frame = tk.Frame(root, bg='white')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

choose_img = tk.Button(root, text='Choose Image', padx=35, pady=10, fg='white', bg='grey', command=load_img)
choose_img.pack(side=tk.LEFT)
predict_btn = tk.Button(root, text='Predict Emotion', padx=35, pady=10, fg='white', bg='grey', command=classify_img)
predict_btn.pack(side=tk.RIGHT)

classifier = load_model('EmotionModel.h5')
root.mainloop()
