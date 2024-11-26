import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

def build_generator():
    model = Sequential()
    model.add(Dense(256 * 4 * 4, input_dim=100))  # Correctly shape the dense layer output
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))  # Reshape output to match input shape of Conv2DTranspose layer
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='tanh'))
    return model

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=2, input_shape=(32, 32, 1), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

generator = build_generator()

discriminator.trainable = False
combined = Sequential([generator, discriminator])
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

def train_gan(epochs, batch_size=128, save_interval=100):
    (X_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    X_train = (X_train - 127.5) / 127.5  # Rescale to [-1, 1]
    X_train = np.expand_dims(X_train, axis=3)
    
    # Resize images to (32, 32, 1)
    X_train = tf.image.resize(X_train, (32, 32))

    half_batch = int(batch_size / 2)

    for epoch in range(epochs):
        # Ensure idx is a list of integers
        idx = np.random.choice(X_train.shape[0], half_batch, replace=False).tolist()
        real_images = tf.gather(X_train, idx)

        noise = np.random.normal(0, 1, (half_batch, 100))
        fake_images = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_images, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.array([1] * batch_size)
        g_loss = combined.train_on_batch(noise, valid_y)

        print(f"{epoch + 1}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

        if epoch % save_interval == 0:
            save_images(epoch)

def save_images(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, (examples, 100))
    gen_images = generator.predict(noise)
    gen_images = 0.5 * gen_images + 0.5

    fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize)
    cnt = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            axs[i, j].imshow(gen_images[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"gan_generated_image_{epoch}.png")
    plt.close()

train_gan(epochs=3000, batch_size=64, save_interval=200)