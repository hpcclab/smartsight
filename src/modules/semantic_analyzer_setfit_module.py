import os
from setfit import SetFitModel
from .ai_module_base import BaseAIModel

class SemanticAnalyzerSetfitModule(BaseAIModel):
    """
    Module responsible for analyzing text using SetFit models (Urgency and MoE).
    Replaces the functionality of junk/UseUrgencyModel.py and handles routing logic.
    """
    def __init__(self):
        super().__init__("semantic_analyzer")
        self.urgency_model_path = self.config.get("urgency_model_path", "models/setfit_models/urgency_model")
        self.moe_model_path = self.config.get("moe_model_path", "models/setfit_models/moe_model")
        self.urgency_threshold = float(self.config.get("urgency_threshold", 0.5))
        
        self.urgency_model = None
        self.moe_model = None
        
        # MoE labels based on train_setfit_MoE.py
        self.id2label = {0: "Object", 1: "Face", 2: "OCR", 3: "Other"}

    def load_model(self):
        """Loads both SetFit classifiers from disk."""
        # Load Urgency Model
        if os.path.exists(self.urgency_model_path):
            self.logger.info(f"Loading SetFit Urgency model from: {self.urgency_model_path}")
            try:
                self.urgency_model = SetFitModel.from_pretrained(self.urgency_model_path)
                self.logger.info("SetFit Urgency Model loaded successfully.")
            except Exception as e:
                self.logger.error(f"Failed to load SetFit Urgency model: {e}")
        else:
            self.logger.warning(f"Urgency model directory not found: {self.urgency_model_path}")

        # Load MoE Model
        if os.path.exists(self.moe_model_path):
            self.logger.info(f"Loading SetFit MoE model from: {self.moe_model_path}")
            try:
                self.moe_model = SetFitModel.from_pretrained(self.moe_model_path)
                self.logger.info("SetFit MoE Model loaded successfully.")
            except Exception as e:
                self.logger.error(f"Failed to load SetFit MoE model: {e}")
        else:
            self.logger.warning(f"MoE model directory not found: {self.moe_model_path}")

    def run_inference(self, input_data: str, **kwargs):
        """
        Classifies the text using the specified loaded model.
        
        Args:
            input_data (str): The text to classify.
            kwargs:
                model_type (str): "urgency" or "moe". Defaults to "urgency".
            
        Returns:
            For urgency: tuple (Category (str), Passed_Threshold (bool), Score (float))
            For moe: str (Category name)
        """
        model_type = kwargs.get("model_type", "urgency")

        if model_type == "urgency":
            if not self.urgency_model:
                self.logger.error("Error: Urgency Model not loaded. Call load_model() first and ensure model path exists.")
                return None

            # Predict probabilities: Returns [[prob_class_0, prob_class_1]]
            # Class 0 = Non-Urgent, Class 1 = Urgent
            probs = self.urgency_model.predict_proba([input_data])[0]
            
            non_urgent_score = float(probs[0])
            urgent_score = float(probs[1])

            self.logger.debug(f"Urgency Scores -> Non-Urgent: {non_urgent_score:.4f} | Urgent: {urgent_score:.4f}")

            if urgent_score >= self.urgency_threshold:
                # It is Urgent
                return ("urgent", True, urgent_score)
            else:
                # It is Non-Urgent
                return ("nonurgent", True, urgent_score)

        elif model_type == "moe":
            if not self.moe_model:
                self.logger.error("Error: MoE Model not loaded. Call load_model() first and ensure model path exists.")
                return None
            
            # Predict returns the class ID directly for MoE
            preds = self.moe_model.predict([input_data])
            label_id = int(preds[0])
            category_name = self.id2label.get(label_id, "Unknown")
            
            self.logger.debug(f"MoE Prediction -> Category: {category_name} (ID: {label_id})")
            return category_name
            
        else:
            self.logger.error(f"Unknown model_type: {model_type}")
            return None
