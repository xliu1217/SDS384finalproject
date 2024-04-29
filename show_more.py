import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Provided extract_features function
def extract_features(img_paths, base_model):
    img_arrays = [image.img_to_array(image.load_img(img_path, target_size=(224, 224))) for img_path in img_paths]
    img_dims = np.array(img_arrays)
    img_preprocess = preprocess_input(img_dims)

    inception_model = base_model.get_layer('inception_v3')
    feature_model = Model(inputs=inception_model.input, 
                          outputs=inception_model.get_layer('mixed4').output)

    features = feature_model.predict(img_preprocess)
    return features.reshape(features.shape[0], -1)

# Load the model
model = load_model('/home/zhutou/project/xai_visualization/previous/fine_tuned_model100.h5')
train_dir = '/home/zhutou/project/xai_visualization/dataset/train'

def most_similar(image_path, top_n, model, train_dir):
    # Extract features of the input image
    input_features = extract_features([image_path], model)

    # List to store similarity scores and image names
    similarity_scores = []

    # Loop through each sub-folder
    for sub_folder in tqdm(os.listdir(train_dir), desc="Processing sub-folders"):
        sub_folder_path = os.path.join(train_dir, sub_folder)

        if os.path.isdir(sub_folder_path):
            img_files = os.listdir(sub_folder_path)
            batch_size = 200  # Adjust as needed

            # Process images in batches
            for i in tqdm(range(0, len(img_files), batch_size), leave=False):
                batch_paths = [os.path.join(sub_folder_path, img_files[j]) for j in range(i, min(i + batch_size, len(img_files)))]
                batch_features = extract_features(batch_paths, model)
                batch_similarities = cosine_similarity(input_features, batch_features).flatten()

                # Store similarity scores with image names
                for j, score in enumerate(batch_similarities):
                    full_image_path = os.path.join(sub_folder, img_files[i+j])
                    similarity_scores.append((full_image_path, score))

    # Sort by similarity score in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Return top N similar images with their scores
    return similarity_scores[:top_n]

# Example usage
top_n_similar_images = most_similar('/home/zhutou/project/xai_visualization/dataset/train/AMERICAN FLAMINGO/001.jpg', 5, model, train_dir)
for image_name, score in top_n_similar_images:
    print(f"{image_name}: {score}")


