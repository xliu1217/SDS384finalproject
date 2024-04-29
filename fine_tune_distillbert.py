from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
import numpy as np
import os

# Disable wandb
os.environ['WANDB_DISABLED'] = 'True'

# Load dataset
df = pd.read_csv('/home/zhutou/project/xai_visualization/previous/xai_dataset.csv')
label_to_int = {label: i for i, label in enumerate(df['label'].unique())}
df['label'] = df['label'].map(label_to_int)
train_df, test_df = train_test_split(df, test_size=0.2)
print(train_df.head())

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize the data
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=512)

train_encodings = tokenize_function(train_df['question'].tolist())
test_encodings = tokenize_function(test_df['question'].tolist())

class XAIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = XAIDataset(train_encodings, train_df['label'].tolist())
test_dataset = XAIDataset(test_encodings, test_df['label'].tolist())

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(df['label'].unique()))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Calculate the number of steps per epoch (assuming your dataset can provide the length)
steps_per_epoch = len(train_dataset)//8
logging_steps = max(1, steps_per_epoch)

training_args = TrainingArguments(
    output_dir='/home/zhutou/project/xai_visualization/previous/results',  
    num_train_epochs=18,              
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=64,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=steps_per_epoch, 
    learning_rate = 5e-5,  
    evaluation_strategy="epoch",     
    save_strategy="epoch",           # Added this line
    load_best_model_at_end=True,     
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
     compute_metrics=compute_metrics
)


trainer.train()

trainer.evaluate()

model.save_pretrained('/home/zhutou/project/xai_visualization/previous/fine-tuned_model_xai_questions')
