import os
import lime
from lime import lime_image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

print("before load")
# Load the model
model = load_model('/home/zhutou/project/xai_visualization/previous/fine_tuned_model100.h5', compile=False)
print("after load")

# Create a LIME Image Explainer
explainer = lime_image.LimeImageExplainer()

# Directory containing the images
input_dir = '/home/zhutou/project/xai_visualization/dataset/examples'
output_dir = '/home/zhutou/project/xai_visualization/lime_image_description/test'

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get all filenames in the input directory and sort them
filenames = sorted(os.listdir(input_dir), key=lambda x: int(x.split('.')[0]))

# Iterate over each file in the input directory
for filename in filenames:
    # Construct the full file path
    img_path = os.path.join(input_dir, filename)

    # Check if it's a file and not a directory
    if os.path.isfile(img_path):
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Explain the prediction
        explanation = explainer.explain_instance(img[0], model.predict, top_labels=3, hide_color=0, num_samples=1000)

        # Get image and mask for the top explanation
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=3, hide_rest=True)

        # Plotting
        plt.figure(figsize=(10, 5))

        # Plot the original image
        plt.subplot(1, 2, 1)
        plt.imshow(img[0] / 2 + 0.5)  # Rescale the image values to [0, 1]
        plt.title("Original Image")

        # Plot the LIME explanation
        plt.subplot(1, 2, 2)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.title("LIME Explanation")

        # Save the plot
        output_filepath = os.path.join(output_dir, f'{filename}')
        plt.savefig(output_filepath)
        plt.close()  # Close the figure to free up memory
