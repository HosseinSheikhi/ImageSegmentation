import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, MaxPooling2D, concatenate
import LoadData

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BATCH_SIZE = 4
IMAGE_SIZE = 256
BUFFER_SIZE = 15000
AUTOTUNE = tf.data.experimental.AUTOTUNE
SEED = 25
N_CHANNELS = 3
N_CLASSES = 2
EPOCHS = 3

dataset = LoadData.LoadData("/home/hossein/synthesisData/training/images/*.png",
                            "/home/hossein/synthesisData/validation/images/*.png",
                            IMAGE_SIZE, BATCH_SIZE, shuffle_buffer_size=5000, seed=123).get_dataset()
print(dataset['train'])
print(dataset['val'])


def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        img = tf.keras.preprocessing.image.array_to_img(display_list[i])
        plt.imshow(img)
        plt.axis('off')
    plt.show()


for image, segmented_mask in dataset['train'].take(1):
    sample_image, sample_mask = image, segmented_mask

display_sample([sample_image[0], sample_mask[0]])


def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    """Return a filter mask with the top 1 predictions
    only.

    Parameters
    ----------
    pred_mask : tf.Tensor
        A [IMG_SIZE, IMG_SIZE, N_CLASS] tensor. For each pixel we have
        N_CLASS values (vector) which represents the probability of the pixel
        being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
        has been predicted class 2 with a probability of 100%.

    Returns
    -------
    tf.Tensor
        A [IMG_SIZE, IMG_SIZE, 1] mask with top 1 predictions
        for each pixels.
    """
    # pred_mask -> [IMG_SIZE, SIZE, N_CLASS]
    # 1 prediction for each class but we want the highest score only
    # so we use argmax
    pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask becomes [IMG_SIZE, IMG_SIZE]
    # but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask


def show_predictions(dataset, num=1):
    """Show a sample prediction.

    Parameters
    ----------
    dataset : [type], optional
        [Input dataset, by default None
    num : int, optional
        Number of sample to show, by default 1
    """

    for image, segmented_mask in dataset.take(num):
        sample_image, sample_mask = image, segmented_mask

        # The model is expecting a tensor of the size
        # [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3]
        # but sample_image[0] is [IMG_SIZE, IMG_SIZE, 3]
        # and we want only 1 inference to be faster
        # so we add an additional dimension [1, IMG_SIZE, IMG_SIZE, 3]
        one_img_batch = sample_image[0][tf.newaxis, ...]
        pred_mask = model.predict(one_img_batch)
        mask = create_mask(pred_mask)
        display_sample([sample_image[0], sample_mask[0], mask[0]])


input_size = (IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS)

initializer = 'he_normal'

# -- Encoder -- #
# Block encoder 1
inputs = Input(shape=input_size)
conv_enc_1 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
conv_enc_1 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_1)

# Block encoder 2
max_pool_enc_2 = MaxPooling2D(pool_size=(2, 2))(conv_enc_1)
conv_enc_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(max_pool_enc_2)
conv_enc_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_2)

# Block  encoder 3
max_pool_enc_3 = MaxPooling2D(pool_size=(2, 2))(conv_enc_2)
conv_enc_3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(max_pool_enc_3)
conv_enc_3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_3)
conv_enc_3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_3)

# Block  encoder 4
max_pool_enc_4 = MaxPooling2D(pool_size=(2, 2))(conv_enc_3)
conv_enc_4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(max_pool_enc_4)
conv_enc_4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_4)
conv_enc_4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_enc_4)
# -- Encoder -- #

# ----------- #
maxpool = MaxPooling2D(pool_size=(2, 2))(conv_enc_4)
conv = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=initializer)(maxpool)
conv = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv)
# ----------- #

# -- Decoder -- #
# Block decoder 1
up_dec_1 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer=initializer)(
    UpSampling2D(size=(2, 2))(conv))
merge_dec_1 = concatenate([conv_enc_4, up_dec_1], axis=3)
conv_dec_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(merge_dec_1)
conv_dec_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_1)
conv_dec_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_1)

# Block decoder 2
up_dec_2 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=initializer)(
    UpSampling2D(size=(2, 2))(conv_dec_1))
merge_dec_2 = concatenate([conv_enc_3, up_dec_2], axis=3)
conv_dec_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(merge_dec_2)
conv_dec_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_2)
conv_dec_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_2)

# Block decoder 3
up_dec_3 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=initializer)(
    UpSampling2D(size=(2, 2))(conv_dec_2))
merge_dec_3 = concatenate([conv_enc_2, up_dec_3], axis=3)
conv_dec_3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(merge_dec_3)
conv_dec_3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_3)

# Block decoder 4
up_dec_4 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=initializer)(
    UpSampling2D(size=(2, 2))(conv_dec_3))
merge_dec_4 = concatenate([conv_enc_1, up_dec_4], axis=3)
conv_dec_4 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer=initializer)(merge_dec_4)
conv_dec_4 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_4)
# conv_dec_4 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer=initializer)(conv_dec_4)
# -- Decoder -- #

output = Conv2D(N_CLASSES, 1, activation='softmax')(conv_dec_4)

model = tf.keras.Model(inputs=inputs, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, epsilon=1e-6),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# for image, mask in dataset['train'].take(1):
#     sample_image, sample_mask = image, mask

show_predictions(dataset['val'], 1)


STEPS_PER_EPOCH = 15000 // BATCH_SIZE
VALIDATION_STEPS = 7449 // BATCH_SIZE

# On GPU
model_history = model.fit(dataset['train'], epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=dataset['val'])

show_predictions(dataset['val'], num=10)

model.save("/home/hossein/imageSegmentation/lessImage")
model.save_weights("/home/hossein/imageSegmentation/lessImage/weights")
