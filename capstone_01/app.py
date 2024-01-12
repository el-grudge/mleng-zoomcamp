import streamlit as st
from PIL import Image
import requests
import boto3

# Load secrets from secrets.toml
url = st.secrets["aws"]["eks_url"]
s3_bucket_name = st.secrets["aws"]["s3_bucket_name"]
image_url = st.secrets["aws"]["image_url"]

st.title("🌍Location Classifier App")

st.header('Upload an image of your location')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_column_width=False, width=300)

    # Save the image to a file
    image.save("temp.jpg", format="JPEG")

    s3 = boto3.client('s3')
    with open("temp.jpg", "rb") as data:
        s3.upload_fileobj(data, f'{s3_bucket_name}', 'myimage.jpg', ExtraArgs={'ACL': 'public-read'})

    # Now 'myimage.jpg' is accessible
    img_url = f'{image_url}'

    # Send the image URL as JSON data
    result = requests.post(url, json={"url": img_url})
    st.write("Prediction result:", result.json())