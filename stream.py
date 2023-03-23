import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from random import randint
from glob import glob

def get_images(data, clas):
    return [cv2.imread(i) for i in glob('streamlit_samples/'+data+'/'+clas+'*')]

mnist = load_model('models/mnist.h5')
cifar10 = load_model('models/cifar10.h5')
cifar100 = load_model('models/cifar100.h5')
fmnist = load_model('models/fmnist.h5')
st.markdown('<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

option = st.selectbox(
    'Which model do you like to use model?',
    ('mnist', 'cifar10', 'cifar100', 'fashion mnist'))

######################
######   for MNIST  #######
######################

if option == "mnist":
    st.title('MNIST Digit Recognizer')

    st.header(":red[Sample images for classes]")
    clas = st.radio(
    "Choose class",
    ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'), horizontal=True)
    images = get_images(option, clas)
    rand = randint(0, 9)
    a = cv2.resize(images[rand], (112,112), interpolation = cv2.INTER_AREA)
    st.image(a)

    st.header(":blue[Using model]")
    
    genre = st.radio(
    "Choose to use model",
    ('Draw by hand', 'Upload image'))

    if genre == 'Draw by hand':
        

        st.markdown('''
        Try to write a digit!
        ''')

        SIZE = 192
        canvas_result = st_canvas(
            fill_color='#000000',
            stroke_width=20,
            stroke_color='#FFFFFF',
            background_color='#000000',
            width=SIZE,
            height=SIZE,
            drawing_mode="freedraw",
            key='canvas')

        if canvas_result.image_data is not None:
            img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
            rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            st.write('Model Input')
            st.image(rescaled)
    else:
        img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            img_array = np.array(image)


    if st.button('Predict'):
        try:
            test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            val = mnist.predict(test_x.reshape(1, 28, 28))
            st.write(f'result: {np.argmax(val[0])}')
            st.bar_chart(val[0])
        except:
            pass
        try:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            img_array = cv2.resize(img_array.astype('uint8'), (28, 28))
            img_array.reshape(1, 28, 28)
            val = mnist.predict(img_array.reshape(1, 28, 28))
            st.write(f'result: {np.argmax(val[0])}')
            st.bar_chart(val[0])
        except:
            pass


#######################
######   for CIFAR10  #######
#######################

elif option == "cifar10":
    st.title('CIFAR10 Recognizer')
    classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    st.header(":white[Sample images for classes]")

    clas = st.radio(
    "Choose class",
    classes, horizontal=True)
    images = get_images(option, clas)
    rand = randint(0, 9)
    a = cv2.resize(images[rand], (128,128), interpolation = cv2.INTER_AREA)
    st.image(a)

    st.header(":blue[Using model]")
    
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_array = np.array(image)


    if st.button('Predict'):
        try:
            img_array = cv2.resize(img_array.astype('uint8'), (32, 32))
            img_array = np.expand_dims(img_array, axis=1)
            img_array = img_array.transpose((1,0,2,3))
            val = cifar10.predict(img_array)
            st.write(f'result: {classes[np.argmax(val[0])]}')
            st.bar_chart(val[0])
        except:
            pass


########################
######   for CIFAR100  #######
########################

elif option == "cifar100":
    st.title('CIFAR100 Recognizer')
    classes = [
           'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge',\
           'bus','butterfly','camel','can','castle','caterpillar','cattle','chair','chimpanzee','clock','cloud',\
           'cockroach','couch','cra','crocodile','cup','dinosaur','dolphin','elephant','flatfish','forest','fox','girl',\
           'hamster','house','kangaroo','keyboard','lamp','lawn_mower','leopard','lion','lizard','lobster','man',\
           'maple_tree','motorcycle','mountain','mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree',\
           'pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road',\
           'rocket','rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel',\
           'streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor','train',\
           'trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm',
]
    st.header(":white[Sample images for classes]")

    clas = st.radio(
    "Choose class",
    classes, horizontal=True)
    images = get_images(option, clas)
    rand = randint(0, 9)
    a = cv2.resize(images[rand], (112,112), interpolation = cv2.INTER_AREA)
    st.image(a)

    st.header(":blue[Using model]")
    
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_array = np.array(image)


    if st.button('Predict'):
        try:
            img_array = cv2.resize(img_array.astype('uint8'), (32, 32))
            img_array = np.expand_dims(img_array, axis=1)
            img_array = img_array.transpose((1,0,2,3))
            val = cifar100.predict(img_array)
            st.write(f'result: {classes[np.argmax(val[0])]}')
            st.bar_chart(val[0])
        except:
            pass


#############################
######   for FASHION MNIST  #######
#############################

else:
    st.title('FASHION MNIST Recognizer')

    st.header(":red[Sample images for classes]")
    classes = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    clas = st.radio(
    "Choose class",
    classes, horizontal=True)
    images = get_images('fmnist', clas)
    rand = randint(0, 9)
    a = cv2.resize(images[rand], (112,112), interpolation = cv2.INTER_AREA)
    st.image(a)

    st.header(":blue[Using model]")
    
    genre = st.radio(
    "Choose to use model",
    ('Draw by hand', 'Upload image'))

    if genre == 'Draw by hand':
        

        st.markdown('''
        Try to write a digit!
        ''')

        SIZE = 192
        canvas_result = st_canvas(
            fill_color='#000000',
            stroke_width=20,
            stroke_color='#FFFFFF',
            background_color='#000000',
            width=SIZE,
            height=SIZE,
            drawing_mode="freedraw",
            key='canvas')

        if canvas_result.image_data is not None:
            img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
            rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            st.write('Model Input')
            st.image(rescaled)
    else:
        img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            img_array = np.array(image)


    if st.button('Predict'):
        try:
            test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            val = mnist.predict(test_x.reshape(1, 28, 28))
            st.write(f'result: {classes[np.argmax(val[0])]}')
            st.bar_chart(val[0])
        except:
            pass
        try:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            img_array = cv2.resize(img_array.astype('uint8'), (28, 28))
            img_array.reshape(1, 28, 28)
            val = fmnist.predict(img_array.reshape(1, 28, 28))
            st.write(f'result: {classes[np.argmax(val[0])]}')
            st.bar_chart(val[0])
        except:
            pass
