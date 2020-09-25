from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet',include_top=True)
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)


#print(model.summary())
#plot_model(model, to_file='vgg.png')


img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = base_model.predict(x)

#print('Predicted:', decode_predictions(block4_pool_features, top=5)[0])

#print out the result of the prediction
# convert the probabilities to class labels
label = decode_predictions(block4_pool_features)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))

