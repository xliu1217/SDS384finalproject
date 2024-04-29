import shap
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json

tf.config.experimental.enable_tensor_float_32_execution(False)

gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load your trained model
model = load_model('/home/zhutou/project/xai_visualization/previous/fine_tuned_model100.h5', compile=False)
# Now compile the model with a standard Adam optimizer
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

def model_pred(image_raw):
    tmp = image_raw.copy()
    # image_processed = preprocess_input(tmp)
    predicted = model.predict(tmp)
    return predicted

# Load and preprocess your images
def load_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return img_array

# Replace with paths to your test images
test_image_paths = []
for i in range (200):
    test_image_paths.append('/home/zhutou/project/xai_visualization/dataset/examples/{:03d}.jpg'.format(i+1))

test_images_raw = np.array([load_img(path) for path in test_image_paths])
json_file = open("/home/zhutou/project/xai_visualization/previous/class_indices.json")
class_json = json.load(json_file)
json_file.close()
class_name = []
for key in class_json.keys():
    class_name.append(key)

masker_blur = shap.maskers.Image("blur(128,128)", test_images_raw[0].shape)
# Initialize the SHAP explainer
explainer = shap.Explainer(model_pred, masker_blur, output_names=class_name)

full_shap_values = explainer(test_images_raw[32:33], max_evals=5000, batch_size=500, outputs=shap.Explanation.argsort.flip[:2])
fig, ax = plt.subplots()
shap.image_plot(full_shap_values)
plt.savefig('/home/zhutou/project/xai_visualization/previous/shap_partition.png')
plt.close()

# for i in range(200):
#     full_shap_values = explainer(test_images_raw[i:i+1], max_evals=5000, batch_size=500, outputs=shap.Explanation.argsort.flip[:1])
#     fig, ax = plt.subplots()
#     shap.image_plot(full_shap_values)
#     plt.savefig('/home/zhutou/project/xai_visualization/previous/shap_partition_result/shap_partition_{:03d}.png'.format(i+1))
#     plt.close()