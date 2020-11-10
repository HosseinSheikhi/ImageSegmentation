import tensorflow as tf
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
reconstructed_model = tf.keras.models.load_model("/home/hossein/imageSegmentation/lessImage")


def parse_image(image_path: str) -> dict:
    # load the raw data from the file as a string
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.uint8)
    return {'image': img}


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


# test_dataset = tf.data.Dataset.list_files("/home/hossein/dev_ws/*.png")
# test_dataset = test_dataset.map(parse_image)
img = tf.io.read_file("/home/hossein/dev_ws/first_overhead_cam.jpg")
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.convert_image_dtype(img, tf.uint8)
img = tf.cast(img, tf.float32) / 255.0
img = tf.image.resize(img, (128, 128))
img = tf.keras.backend.expand_dims(img, 0)

pred = reconstructed_model.predict(img)

pred_mask = tf.argmax(pred, axis=-1)
# pred_mask becomes [IMG_SIZE, IMG_SIZE]
# but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
pred_mask = tf.expand_dims(pred_mask, axis=-1)

pred_img = tf.keras.preprocessing.image.array_to_img(pred_mask[0])
plt.imshow(pred_img)
plt.show()