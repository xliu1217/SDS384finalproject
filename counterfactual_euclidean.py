import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
from PIL import Image as PILImage  # Importing PIL Image with an alias

# Provided extract_features function
def extract_features(img_paths, base_model):
    img_arrays = [image.img_to_array(image.load_img(img_path, target_size=(224, 224))) for img_path in img_paths]
    img_dims = np.array(img_arrays) * 1.0 / 255
    # img_preprocess = preprocess_input(img_dims)

    inception_model = base_model.get_layer('inception_v3')
    feature_model = Model(inputs=inception_model.input, 
                          outputs=inception_model.get_layer('mixed4').output)

    features = feature_model.predict(img_dims)
    return features.reshape(features.shape[0], -1)

# Load the model
# Load the model without its optimizer
model = load_model('/home/zhutou/project/xai_visualization/previous/fine_tuned_model100.h5', compile=False)
print("Model loaded without optimizer.")

# Now compile the model with a standard Adam optimizer
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
print("Model compiled with standard Adam optimizer.")
train_dir = '/home/zhutou/project/xai_visualization/dataset/train'

def most_similar(image_path, bird_type, top_n, model, train_dir, save_dir):
    # Extract the class name of the input image
    input_class = bird_type  # Use the bird_type parameter instead of deriving from image path

    # Extract features of the input image
    input_features = extract_features([image_path], model)

    # List to store similarity scores and image names
    similarity_scores = []

    # Loop through each sub-folder
    for sub_folder in tqdm(os.listdir(train_dir), desc="Processing sub-folders"):
        sub_folder_path = os.path.join(train_dir, sub_folder)

        if os.path.isdir(sub_folder_path) and sub_folder != input_class:  # Skip the input image's class based on bird_type
            img_files = os.listdir(sub_folder_path)
            batch_size = 2000  # Adjust as needed

            # Process images in batches
            for i in tqdm(range(0, len(img_files), batch_size), leave=False):
                batch_paths = [os.path.join(sub_folder_path, img_files[j]) for j in range(i, min(i + batch_size, len(img_files)))]
                batch_features = extract_features(batch_paths, model)
                batch_distances = euclidean_distances(input_features, batch_features).flatten()

                # Store similarity scores with image names
                for j, distance in enumerate(batch_distances):
                    full_image_path = os.path.join(sub_folder, img_files[i+j])
                    similarity_scores.append((full_image_path, distance))

    # Sort by similarity score in ascending order (since it's distance)
    similarity_scores.sort(key=lambda x: x[1])

    # Save top N similar images and print their paths with scores
    for i, (image_name, score) in enumerate(similarity_scores[:top_n]):
        print(f"{image_name}: {score}")
        pil_image = PILImage.open(os.path.join(train_dir, image_name))
        only_image_file_name = os.path.basename(image_name)
        save_path = os.path.join(save_dir, f"top_{i+1}_{only_image_file_name}")
        pil_image.save(save_path)  # Use pil_image instead of image

    # Return top N similar images with their scores from different classes
    return similarity_scores[:top_n]

# Example usage
save_dir = '/home/zhutou/project/xai_visualization/previous/counter_results'  # Define the directory where you want to save the images
top_n_similar_images = most_similar('/home/zhutou/project/xai_visualization/dataset/examples/190.jpg', "MALLARD DUCK", 1, model, train_dir, save_dir)
