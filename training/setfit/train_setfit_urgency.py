import sys
import os
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers import SentenceTransformer
import logging
from sentence_transformers import LoggingHandler
import json

# Add root folder to python path to access config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.config import get_config
config = get_config().get("semantic_analyzer", {})
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

with open(config.get("urgency_setfit_train_dataset", "urgentTrainSetFit.json"), "r") as f:
    data = json.load(f)

urgentRequests = data["urgent"]
nonUrgentRequests = data["nonurgent"]

labels = []

for i in urgentRequests:
    labels.append(1)
for i in nonUrgentRequests:
    labels.append(0)

data = {
    "text": urgentRequests + nonUrgentRequests,
    "label": labels
}

dataset = Dataset.from_dict(data)

print("Loading Model...")
model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
print("Model Loaded.")


args = TrainingArguments(
    batch_size=16,
    num_epochs=1,
    num_iterations=20,
    loss=CosineSimilarityLoss 
)

# 4. Initialize Trainer
# CHANGE: removed 'loss_class' from here
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    metric="accuracy",
)

# 5. Train
print("Training model...")
trainer.train()

# 6. Test
test_sentences = [
    "I need to know what is in front of me right now.",
    "What is in front of me?",
    "This is a total disaster, help.",
    "Who do you think he is?",
    "Quickly describe my surroundings."
]

print("\n--- Inference Results ---")
preds = model(test_sentences)

for sentence, label in zip(test_sentences, preds):
    status = "🔴 URGENT" if label == 1 else "🟢 Normal"
    print(f"{status} | {sentence}")

# 7. Save
save_path = config.get("urgency_model_path", "models/setfit_models/urgency_model")

# Ensure base directory exists just in case
save_dir = os.path.dirname(save_path)
if save_dir and not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

model.save_pretrained(save_path)
print(f"Model saved to '{save_path}'")