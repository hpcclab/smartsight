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
            stream=True
        )
        
        # Check if the response is a streaming generator/iterator
        if response and not isinstance(response, str) and (hasattr(response, "__next__") or hasattr(response, "__iter__")):
            token_list = []
            self.global_response_module.add_message(token_list, priority=10)
            full_response = []
            for token in response:
                token_list.append(token)
                full_response.append(token)
            token_list.append(None)  # Sentinel to signal completion
            return "".join(full_response)
        else:
            self.global_response_module.add_message(response or "Failed to get response.", priority=10)
            if response:
                return response
            else:
                return "Failed to get response from the model."

