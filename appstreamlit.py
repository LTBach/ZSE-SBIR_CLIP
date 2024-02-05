import os
import time
import random

import torch
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from utils.util import get_output_file, load_checkpoint
from options import Option
from model.model import Model
from retrieval import retrieve


prev_color = '#000'
prev_width = 3
PATH = os.path.join('datasets', "Sketchy", "256x256", "photo", "tx_000000000000_ready")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = Option().parse()
    
# Setup model
model = Model(args)
for param in model.parameters():
    param.requires_grad = False

model.eval()
model.to(device)
        
if os.path.isfile(args.load):
    checkpoint = load_checkpoint(args.load)
    cur = model.state_dict()
    new = {k: v for k, v in checkpoint['model'].items() if k in cur.keys()}
    cur.update(new)
    model.load_state_dict(cur)  
else:
    raise ImportError("Pre-trained weigths not found!")
    
def main(args):
    # Specify canvas parameters in application
    global prev_color
    global prev_width

    num_retrieve = st.sidebar.slider('Number of retrieved images:', 3, 15, 12, 3)

    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
    )
    
    mode = st.sidebar.radio(
        "Choose mode",
        ["Pencil", "Eraser"])

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == 'point':
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
        
        
    
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    #bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    
    if mode == "Eraser":
        stroke_color = bg_color
        stroke_width += 20
        

    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        #background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=600,
        width=600,
        drawing_mode=drawing_mode,
        #display_toolbar=False,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
    )

    search_button = st.button("Search")#, on_click = retrieve_bt, args = (canvas_result.image_data))
    #current_path = os.getcwd()
    #result_path = os.path.join(current_path, args.output_dir)

    embeded = np.load('storage/all.npz')
    df = pd.read_csv('storage/cluster.csv')
    centroid_feats = torch.from_numpy(np.load('storage/centroid_embed.npy')).to(device)
    
    if search_button:
        if canvas_result.image_data is not None:
            #st.write(type(canvas_result.image_data))
            sk = Image.fromarray(canvas_result.image_data)
            #im.resize((224, 224)).save("sketch.png")
            sk.save("sketch.png")
            # rank_list = retrieve(args, "sketch.png")
            
            # Perform searching
            
            #result_path = os.path.join(result_path, f"results")
            t0 = time.time()
            path_list = retrieve(device, embeded, df, centroid_feats, "sketch.png", "Sketchy", model, False, args, None, k=num_retrieve, t = 3)
            t1 = time.time()
            
            execute_time = t1-t0
            st.write(f"Search time: {execute_time:.2f}s")
            # Show results
            with st.container():
                #img_path = os.path.join(PATH, "airplane")
                #img_list = []
                #imlst = os.listdir(img_path)
                #random.shuffle(imlst)
                #for img in imlst[:num_retrieve]:
                #    img_list.append(Image.open(os.path.join(img_path, img)))

                rows = num_retrieve // 3
                for i in range(rows):
                    cols = st.columns(3)
                    with cols[0]:
                        st.image(path_list[i * 3])
                    with cols[1]:
                        st.image(path_list[i * 3 + 1])
                    with cols[2]:
                        st.image(path_list[i * 3 +2])
                # for path in rank_list:
                #     img_list.append(Image.open(path))
                # st.image(img_list)
 


if __name__ == '__main__':
    
    # Run app
    main(args)
