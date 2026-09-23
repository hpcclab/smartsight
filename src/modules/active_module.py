from .ai_manager import AI_manager
from config.config import get_config

class ActiveModule:
    def __init__(self, global_response_module):
        self.config = get_config().get("openrouter_api", {})
        self.global_response_module = global_response_module
            
    def ProcessRequest(self, text_input: str):
        """
        Process text input by calling the API MLLM through the AI manager
        with image input enabled. Tries each model in model_priority until
        one returns a response.
        """
        models = self.config.get("model_priority") or []
        if not models:
            legacy_model = self.config.get("model")
            models = [legacy_model] if legacy_model else []

        for model_name in models:
            response = AI_manager.execute_module(
                lambda m: m.__class__.__name__ == "APIMLLMModule",
                input_key="active_request",
                input_data=text_input,
                use_image=True,
                model=model_name,
                stream=False
            )
            if response:
                self.global_response_module.add_message(response, priority=10)
                return response

        return "Failed to get response from the model."
