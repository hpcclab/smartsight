from .ai_manager import AI_manager
from .edge_cloud_merger import EdgeCloudMerger
from config.config import get_config

class ActiveModule:
    def __init__(self, global_response_module):
        self.config_cloud = get_config().get("openrouter_api", {})
        self.config_edge = get_config().get("ollama_llm", {})
        self.global_response_module = global_response_module
        self.merger = EdgeCloudMerger()
            
    def ProcessRequest(self, text_input: str, frame=None):
        """
        Process text input by executing Edge-Cloud fusion (Algorithm 1) via the EdgeCloudMerger.
        """
        cloud_model = self.config_cloud.get("model")
        edge_model = self.config_edge.get("model")

        response = self.merger.run_inference(
            text_input=text_input,
            use_image=True,
            cloud_model=cloud_model,
            edge_model=edge_model,
            frame=frame
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


