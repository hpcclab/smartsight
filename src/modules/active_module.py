from .ai_manager import AI_manager
from config.config import get_config

class ActiveModule:
    def __init__(self, global_response_module):
        self.config = get_config().get("openrouter_api", {})
        self.global_response_module = global_response_module
            
    def ProcessRequest(self, text_input: str):
        """
        Process text input by calling the API MLLM through the AI manager
        with image input enabled.
        """
        model_name = self.config.get("model", "gemma3:12b")
        response = AI_manager.execute_module(
            lambda m: m.__class__.__name__ == "APIMLLMModule",
            input_key="active_request",
            input_data=text_input,
            use_image=True,
            model=model_name,
            stream=False
        )
        self.global_response_module.add_message(response, priority=10)
        if response:
            return response
        else:
            return "Failed to get response from the model."
