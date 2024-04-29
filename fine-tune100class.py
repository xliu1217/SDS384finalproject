from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Forces TensorFlow to use only CPU


# Load your fine-tuned model
model = load_model('/home/zhutou/project/xai_visualization/previous/my_fine_tuned_model525.h5', compile=False)
print("Load done")
# Remove the last layer
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

# Add a new output layer for 100 classes
new_output = Dense(100, activation='softmax')(model.output)
model = Model(inputs=model.inputs, outputs=new_output)

# Recompile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare your data using ImageDataGenerator
train_data_dir = '/home/zhutou/project/xai_visualization/dataset/train'
validation_data_dir = '/home/zhutou/project/xai_visualization/dataset/valid'

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical')

# Fine-tune the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)


# Save the fine-tuned model
model.save('/home/zhutou/project/xai_visualization/previous/new_fine_tuned_model100.h5')

# Save class indices to a JSON file
class_indices = train_generator.class_indices
with open('/home/zhutou/project/xai_visualization/previous/class_indices.json', 'w') as file:
    json.dump(class_indices, file)
print("Class indices saved.")