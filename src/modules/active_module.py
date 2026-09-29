from .ai_manager import AI_manager
from .fusion_module import FAILURE_TEXT, FusionModule
from config.config import get_config
from utilities.logging_setup import get_logger
class ActiveModule:
    def __init__(self, global_response_module, fusion=None):
        self.logger = get_logger(self.__class__.__name__)
        full = get_config()
        self.config = full.get("openrouter_api", {})
        self._fusion_enabled = full.get("fusion", {}).get("enabled", True)
        self.global_response_module = global_response_module
        if fusion is not None:
            self.fusion = fusion
        elif self._fusion_enabled:
            self.fusion = FusionModule(global_response_module)
        else:
            self.fusion = None
            
    def ProcessRequest(self, text_input: str):
        """
        Answer an active request. Fusion is the default: the local and cloud
        models run together and speech starts from the local stream.
        With fusion disabled, the cloud model list is tried once and the full
        answer is spoken as a single message.
        """
        if self.fusion is not None and self._fusion_enabled:
            return self.fusion.handle(text_input)
        return self._cloud_only(text_input)

    def _cloud_only(self, text_input: str):
        models = self.config.get("model_priority") or []
        if not models:
            self.logger.warning("No models specified in model_priority. Falling back to legacy model.")
            legacy_model = self.config.get("model")
            models = [legacy_model] if legacy_model else []
        self.logger.info(f"Models to try: {models}")
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
                self.logger.info(f"Success. Model {model_name} response: \n{response}\n")
                self.global_response_module.add_message(response, priority=10, message_expiration=30.0)
                return response
            else:
                self.logger.warning("Error. Retrying with next model.")

        return FAILURE_TEXT
