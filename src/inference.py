import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import torch
import os
from torchvision.transforms import ToTensor,Resize,Normalize,Compose
l=['air hockey',
 'ampute football',
 'archery',
 'arm wrestling',
 'axe throwing',
 'balance beam',
 'barell racing',
 'baseball',
 'basketball',
 'baton twirling',
 'bike polo',
 'billiards',
 'bmx',
 'bobsled',
 'bowling',
 'boxing',
 'bull riding',
 'bungee jumping',
 'canoe slamon',
 'cheerleading',
 'chuckwagon racing',
 'cricket',
 'croquet',
 'curling',
 'disc golf',
 'fencing',
 'field hockey',
 'figure skating men',
 'figure skating pairs',
 'figure skating women',
 'fly fishing',
 'football',
 'formula 1 racing',
 'frisbee',
 'gaga',
 'giant slalom',
 'golf',
 'hammer throw',
 'hang gliding',
 'harness racing',
 'high jump',
 'hockey',
 'horse jumping',
 'horse racing',
 'horseshoe pitching',
 'hurdles',
 'hydroplane racing',
 'ice climbing',
 'ice yachting',
 'jai alai',
 'javelin',
 'jousting',
 'judo',
 'lacrosse',
 'log rolling',
 'luge',
 'motorcycle racing',
 'mushing',
 'nascar racing',
 'olympic wrestling',
 'parallel bar',
 'pole climbing',
 'pole dancing',
 'pole vault',
 'polo',
 'pommel horse',
 'rings',
 'rock climbing',
 'roller derby',
 'rollerblade racing',
 'rowing',
 'rugby',
 'sailboat racing',
 'shot put',
 'shuffleboard',
 'sidecar racing',
 'ski jumping',
 'sky surfing',
 'skydiving',
 'snow boarding',
 'snowmobile racing',
 'speed skating',
 'steer wrestling',
 'sumo wrestling',
 'surfing',
 'swimming',
 'table tennis',
 'tennis',
 'track bicycle',
 'trapeze',
 'tug of war',
 'ultimate',
 'uneven bars',
 'volleyball',
 'water cycling',
 'water polo',
 'weightlifting',
 'wheelchair basketball',
 'wheelchair racing',
 'wingsuit flying']
with st.sidebar:
    image = Image.open(os.getcwd()+'\\'+'src\\sports_icon.png')
    st.image(image)

    st.text("""
            100 Sports Image Classification
            using fine-tuned ResNet18 
            model 
            """
    )
st.title('100 Sports Classification 🏋️‍♀️🤾‍♂️🚴‍♂️')
uploadFile=st.file_uploader("Upload an image of sport")

def load_image(img):
    im = Image.open(img).convert('RGB')
    image = np.array(im)
    return image

def predict(processed_image):
    model = torch.load("models\sports_100_classification_resnet18_adam_10_epochs.pth",map_location=torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        yhat=model(processed_image)
        pred=torch.argmax(yhat,1)
    return yhat, pred
def process_image(img):
    return Compose([ToTensor(),Resize((224,224)),Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])(img).unsqueeze(0)
if uploadFile is not None:
    img = load_image(uploadFile)

    processed_image=process_image(img)

    yhat,pred=predict(processed_image)

    class_score=pd.DataFrame({'sport':l,'score':yhat[0]})


    st.image(np.array(img))
    st.write("Image Uploaded Successfully")
    st.header("Sport = "+l[pred])
    st.title("Class Scores")
    st.bar_chart(class_score,x='sport')
else:
    st.write("Make sure you image is in JPG/PNG Format.")


