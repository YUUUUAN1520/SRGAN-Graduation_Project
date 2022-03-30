import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, UpSampling2D, Flatten, BatchNormalization
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, PReLU, Add
from tensorflow.keras.models import Model


# define Residual block
def res_block(input_layer, kernel_size, filters, strides):
    input_layer_temp = input_layer

    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(input_layer)
    x = BatchNormalization(momentum=0.5)(x)
    x = PReLU(alpha_initializer="zeros", alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(x)
    x = BatchNormalization(momentum=0.5)(x)

    output_layer = Add()([input_layer_temp, x])

    return output_layer

# define Upsamling Block
def up_sampling_block(input_layer, kernel_size, filters, strides):
    x = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(input_layer)
    x = UpSampling2D(size = 2)(x)
    output_layer = LeakyReLU(alpha = 0.2)(x)

    return output_layer
    

# define Convolution block
def conv_block(input_layer, kernel_size, filters, strides):
    
    x = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(input_layer)
    x = BatchNormalization(momentum = 0.5)(x)
    output_layer = LeakyReLU(alpha = 0.2)(x)
    
    return output_layer

# Build Generator
def get_generator(input_shape, show_model=True):
    gen_input = Input(shape=input_shape)
    x = Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(gen_input)
    x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(x)
    
    temp_layer = x 

    # Using 16 res_blocks
    for _ in range(16):
        x = res_block(input_layer=x, kernel_size=3, filters=64, strides=1)
        
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = BatchNormalization(momentum=0.5)(x)
    model = Add()([temp_layer, x])

    # Using 2 up_sampling_block : 4x imgs
    for _ in range(2):
        x = up_sampling_block(input_layer=x, kernel_size=3, filters=128, strides=1)

    x = Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(x)
    gen_output = Activation('tanh')(x)

    model = Model(inputs=gen_input, outputs=gen_output)
    
    if show_model:
        print("Generator : ")
        model.summary()
        
    return model


# Build Discriminator
def get_discriminator(input_shape, show_model=True):
    dis_input = Input(shape = input_shape)
    
    x = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = "same")(dis_input)
    x = LeakyReLU(alpha = 0.2)(x)
    
    x = conv_block(x, 32, 3, 2)
    x = conv_block(x, 64, 3, 1)
    x = conv_block(x, 64, 3, 2)
    x = conv_block(x, 128, 3, 1)
    x = conv_block(x, 128, 3, 2)
    x = conv_block(x, 256, 3, 1)
    x = conv_block(x, 256, 3, 2)
    
    x = Flatten()(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha = 0.2)(x)
   
    x = Dense(1)(x)
    dis_output = Activation('sigmoid')(x) 
    
    model = Model(inputs=dis_input, outputs=dis_output)

    if show_model:
        print("Discriminator : ")
        model.summary()

    return model
    
# combine the generator and discriminator to train generator
def get_gan(Discriminator, Generator, generator_input_shape, vgg_loss):
    Discriminator.trainable = False
    gen_input = Input(shape=generator_input_shape)
    gen_output = Generator(gen_input)
    dis_output = Discriminator(gen_output)

    GAN = Model(inputs=gen_input,outputs=[gen_output, dis_output])

    return GAN