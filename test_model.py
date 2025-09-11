import tensorflow as tf

model = tf.keras.models.load_model("ship.keras")

label_map = {
    'coastguard': 0, 'containership': 1, 'corvette': 2, 'cruiser': 3,
    'cv': 4, 'destroyer': 5, 'ferry': 6, 'methanier': 7,
    'sailing': 8, 'smallfish': 9, 'submarine': 10, 'tug': 11, 'vsmallfish': 12
}

class_names = [name for name, idx in sorted(label_map.items(), key=lambda item: item[1])]

img = tf.keras.utils.load_img("ship.jpeg", target_size=(32, 32))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[tf.argmax(score)], 100 * tf.reduce_max(score))
)