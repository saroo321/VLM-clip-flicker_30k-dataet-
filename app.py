import streamlit as st
import faiss
import numpy as np
import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import pickle

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load FAISS index and image name list
index = faiss.read_index("faiss_index.idx")
image_names = np.load("image_names.npy", allow_pickle=True).tolist()

IMAGE_FOLDER = "flickr30k_images"

st.title("Image Retrieval with CLIP + FAISS")
st.write("Search similar images using text or image query")

# Helper: Get image embedding
def get_image_embedding(image):
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
    return image_features.cpu().numpy().astype(np.float32)

# Helper: Get text embedding
def get_text_embedding(text):
    text_inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    return text_features.cpu().numpy().astype(np.float32)

# Query option
option = st.radio("Choose query type", ["Text", "Image"])

if option == "Text":
    query = st.text_input("Enter your text query:")
    if st.button("Search") and query:
        embedding = get_text_embedding(query)
        D, I = index.search(embedding, k=5)
        st.subheader("Top 5 similar images:")
        for idx in I[0]:
            image_path = os.path.join(IMAGE_FOLDER, image_names[idx])
            st.image(image_path, width=250)

elif option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        query_image = Image.open(uploaded_file).convert("RGB")
        st.image(query_image, caption="Query Image", width=300)

        if st.button("Search"):
            embedding = get_image_embedding(query_image)
            D, I = index.search(embedding, k=5)
            st.subheader("Top 5 similar images:")
            for idx in I[0]:
                image_path = os.path.join(IMAGE_FOLDER, image_names[idx])
                st.image(image_path, width=250)
                import os
import numpy as np

image_folder = "flickr30k_images"
image_names = sorted(os.listdir(image_folder))  # Make sure the order matches FAISS index!
np.save("image_names.npy", image_names)
print(f"Saved {len(image_names)} image names.")



