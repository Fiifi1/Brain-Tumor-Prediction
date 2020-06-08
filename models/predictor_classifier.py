'''This predictor_classifier.py class was initially 
    used to test the model at development stage
    It can be used as it is. However, similar methods
    and procedures are available in the app.py
'''

# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.vgg16 import preprocess_input

# #import classifier weight from pretrained model
# classifer = load_model('classifier.h5')

# #Image path
# img_path = './data/validation/no/no 77.jpg'

# #set image size
# img = image.load_img(img_path, target_size=(224, 224))
# #Convert image to one dimensional array
# img_array = image.img_to_array(img)

# img_array = np.expand_dims(img_array, axis=0)
# #preprocess image array
# img_data = preprocess_input(img_array)

# #predict model
# result = classifer.predict(img_data)
# #result [[err, acc]]

# #checker
# if result[0][0] == 1:
#     output = 'Negative: Not a Brain Tumor'
# else:
#     output = 'Positive: Brain Tumor' 

# print(output)