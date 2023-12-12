import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np

IMG_SIZE = 227

st.markdown(
    """
    <style>
        /* Thay đổi kiểu chữ và màu chữ */
        body {
            font-family: 'Arial', sans-serif;
            color: #333;
        }

        /* Thay đổi canh lề của trang web */
        .css-1v7hqsl {
            max-width: 1200px;
            margin: auto;
        }

        /* Thay đổi kiểu chữ và màu chữ cho tiêu đề và nút */
        .css-1kz7hm7, .css-16ptfca button {
            font-family: 'Arial', sans-serif;
            color: #fff; /* Màu chữ trắng */
        }
    </style>
    """,
    unsafe_allow_html=True
)

class_list = {'0': 'Bình thường', '1': 'Viêm Phổi'}

st.title('Dự đoán viêm phổi dựa trên hình ảnh X-quang ngực')

input = open('lrc_xray.pkl', 'rb')
model = pkl.load(input)

st.header('Tải lên ảnh X-quang ngực')
uploaded_file = st.file_uploader("Chọn 1 file ảnh", type=(['png', 'jpg', 'jpeg']))

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh thử')

    if st.button('Dự đoán'):
        image = image.resize((IMG_SIZE*IMG_SIZE*3, 1))
        feature_vector = np.array(image)
        label = str((model.predict(feature_vector))[0])

        st.header('Kết quả')
        st.text(class_list[label])
