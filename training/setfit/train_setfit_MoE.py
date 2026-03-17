import json
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from pathlib import Path
import sys
import os

# Add root folder to python path to access config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.config import get_config

config = get_config().get("semantic_analyzer", {})
json_file = config.get("moe_setfit_train_dataset", "Moe_train_Setfit_v5.json")

with open(json_file, 'r') as f:
    raw_data = json.load(f)

text_list = []
label_list = []

# Map categories to integer labels
# 0: Object, 1: Face, 2: OCR
label_map = {
    "Object": 0,
    "Face": 1,
    "OCR": 2,
    "Other": 3
}

# Inverse map for inference later
id2label = {v: k for k, v in label_map.items()}

# Flatten the JSON lists into training vectors
for category, sentences in raw_data.items():
    if category in label_map:
        label_id = label_map[category]
        text_list.extend(sentences)
        # Create a label entry for every sentence
        label_list.extend([label_id] * len(sentences))

data = {
    "text": text_list,
    "label": label_list
}

dataset = Dataset.from_dict(data)

# 2. Load the model
# SetFit handles multi-class classification automatically based on the labels provided
model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# 3. Define arguments
args = TrainingArguments(
    batch_size=16,
    num_epochs=1,
    num_iterations=20,
    loss=CosineSimilarityLoss 
)

# 4. Initialize Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    metric="accuracy",
)

# 5. Train
print(f"Training model on categories: {list(label_map.keys())}...")
trainer.train()

# 6. Test
test_sentences = [
    "Read the text on that sign.",
    "Who is standing next to the car?",
    "What items are sitting on the desk?",
    "Is there a place to keep my pet orangutan?"
]

print("\n--- Inference Results ---")
preds = model(test_sentences)

for sentence, label in zip(test_sentences, preds):
    # Convert integer prediction back to string category
    category_name = id2label[int(label)]
    print(f"[{category_name}] : {sentence}")

# 7. Save
save_path = config.get("moe_model_path", "models/setfit_models/moe_model")

# Ensure base directory exists just in case
save_dir = os.path.dirname(save_path)
if save_dir and not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

model.save_pretrained(save_path)
print(f"Model saved to '{save_path}'")