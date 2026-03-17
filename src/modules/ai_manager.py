import logging
from .TTS_module import TTSModule
from .object_detection_ai_module import ObjectDetectionAIModule
from .facial_recognition_ai_module import FacialRecognitionAIModule
from .ocr_module import OCRModule
from .speech_to_text_module import SpeechToTextModule
from .edge_mllm_module import EdgeMLLMModule
from .semantic_analyzer_setfit_module import SemanticAnalyzerSetfitModule

class AIManager:
    """
    Manager class responsible for instantiating and handling 
    all AI modules in the system.
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing AIManager and instantiating AI modules...")
        
        # Instantiate all AI modules and hold them in a list
        self.modules = [
            TTSModule(),
            ObjectDetectionAIModule(),
            FacialRecognitionAIModule(),
            OCRModule(),
            SpeechToTextModule(),
            EdgeMLLMModule(),
            SemanticAnalyzerSetfitModule()
        ]

    def get_module(self, condition):
        """
        Retrieves an AI module based on a provided lambda condition.
        Returns the module if found, else None.
        """
        return next((m for m in self.modules if condition(m)), None)

    def execute_module(self, condition, input_key, *args, **kwargs):
        """
        Finds an AI module based on the condition and executes it with the provided arguments.
        """
        module = self.get_module(condition)
        if module:
            return module.execute(input_key, *args, **kwargs)
        else:
            self.logger.error("No matching AI module found for execution.")
            return None

    def load_all_models(self):
        """
        Explicitly triggers the load_model routine for all instantiated AI modules.
        This can be called during startup to ensure all weights and configurations are loaded
        into memory before inference begins.
        """
        self.logger.info("Loading models for all initialized AI modules...")
        
        for module in self.modules:
            try:
                module.load_model()
            except Exception as e:
                self.logger.error(f"Failed to load {module.__class__.__name__} model: {e}")
            
        self.logger.info("Finished loading all AI module models.")

AI_manager = AIManager()