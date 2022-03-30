# define content loss using pretrained VGG19 network referred in the paper


from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

class VGG_LOSS(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape
        
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False
        self.model = model
        
    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
    
        return K.mean(K.square(self.model(y_true) - self.model(y_pred)))

