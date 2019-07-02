from keras.models import load_model, model_from_json
from keras.preprocessing import image
import numpy as np
from keras import backend as K
from pprint import pprint

# dimensions of our images
img_width, img_height = 150, 150

# load the model we saved
with open('model_def.json') as ff:
    model_json=ff.read()
    model= model_from_json(model_json)
model.load_weights('first_try.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print('hi')
# predicting images
img = image.load_img('y.jpg', target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# pprint(x)
# images = np.vstack([x])


# images = np.delete(images, [0], axis=0)
classes = model.predict_proba(x, batch_size=10, verbose=0)
print(classes)
image_files = ['y.jpg', 'elephant.jpg', 'n2.jpg', 'y2.jpg', 'wc.jpg', 'bbc.PNG']
images = np.full(shape=(img_width, img_height, 3), fill_value=1, dtype=np.float32)
images = np.expand_dims(images, axis=0)
for img in image_files:
    print(img)
    img = image.load_img(img, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # pprint(x)
    # images = np.vstack([x])
    images = np.append(images, x, axis = 0)

# images = np.delete(images, [0], axis=0)
classes = model.predict_proba(images, batch_size=10, verbose=0)
print(classes)

# get_intermediate_layer_output = K.function([model.layers[0].input],
#                               [model.layers[14].output])
# layer_output = get_intermediate_layer_output([x])[0]
# print(layer_output)

# import keras.models
# import pickle as pkl
#
# with open('model_def.json') as ff:
#     model_json=ff.read()
#     model=keras.models.model_from_json(model_json)
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# model.load_weights('first_try.h5')
# with open('test_mats.pkl','rb') as ff:
#     X_test=pkl.load(ff)
#
# preds=model.predict(X_test)
