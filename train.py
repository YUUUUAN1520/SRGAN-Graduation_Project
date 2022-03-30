from model import get_generator, get_discriminator, get_gan
from vgg19_loss import VGG_LOSS
from Utils import plot_generated_images, denormalize
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import numpy as np 
np.random.seed(10)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

image_shape = (384,384,3)
downscale_factor = 4
__dataset_name__ = 'DIV2K_train_HR'
dataset_tensor = np.load('./datasets/'+ __dataset_name__ + '_tensor.npz')

scaled_image_shape = (image_shape[0] // downscale_factor, 
                      image_shape[1] // downscale_factor, 
                      image_shape[2])



# define optimizer
def get_optimizer():
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam



def save_model(path, trained_generator, trained_discriminator):
    trained_generator.save(path + '/Generator.h5')
    
    trained_discriminator.trainable = True
    trained_discriminator.save(path + '/Discriminator.h5')
    
    
def load_model(path):
    loss = VGG_LOSS(image_shape)
    Optimizer = get_optimizer()
    
    trained_generator = tf.keras.models.load_model(path + '/Generator.h5', compile=False)
    trained_discriminator = tf.keras.models.load_model(path + '/Discriminator.h5', compile=False)
    trained_discriminator.trainable = True
    
    return [trained_generator, trained_discriminator]


def train(epochs, batch_size, show_interval=5, give_model=None, have_trained_epochs=0):
    # load data
    imgs_hr, imgs_lr = dataset_tensor['imgs_hr'], dataset_tensor['imgs_lr']
    imgs_size = imgs_hr.shape[0]
    
    # define loss function.
    loss = VGG_LOSS(image_shape)
    
    # define optimizer function.
    Optimizer = get_optimizer() 
    
    if give_model == None:  # initialize model.
        Generator = get_generator(scaled_image_shape,show_model=False)
        Discriminator = get_discriminator(image_shape,show_model=False)

    else:   # load model.
        Generator = give_model[0]
        Discriminator = give_model[1]
        
    Generator.compile(loss=loss.vgg_loss, optimizer=Optimizer) 
    Discriminator.compile(loss="binary_crossentropy", optimizer=Optimizer)
    
    GAN = get_gan(Discriminator, Generator, generator_input_shape=scaled_image_shape, vgg_loss=loss.vgg_loss)
    GAN.compile(loss=[loss.vgg_loss,"binary_crossentropy"],
                loss_weights=[1.,1e-3],
                optimizer=Optimizer)
    
    for e in range(1,epochs+1):
        print("-" * 15 + "Epoch " + str(e) + " of " + str(epochs) + "-" * 15)
        sum_GAN_loss = 0
        sum_d_loss = 0
        for _ in tqdm(range(imgs_size // batch_size)):

            # -----------------------
            # training Discriminator
            # -----------------------
            selected_index = np.random.choice(imgs_size,batch_size)
            batch_hr, batch_lr = imgs_hr[selected_index], imgs_lr[selected_index]
            
            fake_hr = Generator.predict(batch_lr)
            
            real_labels = np.ones(batch_size)
            fake_labels = np.zeros(batch_size)
            
            Discriminator.trainable = True    
            d_loss_real = Discriminator.train_on_batch(batch_hr, real_labels)
            d_loss_fake = Discriminator.train_on_batch(fake_hr, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # --------------------
            # training Generator
            # --------------------
            selected_index = np.random.choice(imgs_size,batch_size)
            batch_hr, batch_lr = imgs_hr[selected_index], imgs_lr[selected_index]
            
            _labels_GAN = np.ones(batch_size)
            
            Discriminator.trainable = False
            g_loss = GAN.train_on_batch(batch_lr, [batch_hr, _labels_GAN])
            
            sum_d_loss += d_loss
            sum_GAN_loss += g_loss[0]
            
            
        # print average loss 
        print("discriminator_loss : ", sum_d_loss / float(imgs_size // batch_size))
        print("GAN_loss : ", sum_GAN_loss / float(imgs_size // batch_size))
        
        # every show_interval, show(save) img that generator makes.
        if e % show_interval == 0:
            a = np.random.choice(imgs_size)
            hr_test_img, lr_test_img = np.array([imgs_hr[a]]), np.array([imgs_lr[a]])
            plot_generated_images('./result', have_trained_epochs+e, Generator, hr_test_img, lr_test_img)
        
        # every ten epochs, save model.
        if e % 100 == 0:
            save_model('./Trained_model', Generator, Discriminator)
            
    return Generator, Discriminator
            



if __name__ == '__main__':
    trained_generator, trained_discriminator = train(epochs=3000,
                                                     batch_size=4,
                                                     show_interval=5)
    """
    ,
                                                     give_model=model,
                                                     have_trained_epochs=10)
    """






