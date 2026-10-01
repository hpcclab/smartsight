import re

from .fusion_module import FAILURE_TEXT, FusionModule
from config.config import get_commands_config, get_config
from utilities.logging_setup import get_logger

READ_ERROR_TEXT = "Error reading text."


def _rule_matches(rule: str, query: str) -> bool:
    """True when query contains rule as a keyword or the rule regex matches."""
    if rule.casefold() in query.casefold():
        return True
    try:
        return re.search(rule, query, re.IGNORECASE) is not None
    except re.error:
        return False


class ActiveModule:
    def __init__(self, global_response_module, fusion=None):
        self.logger = get_logger(self.__class__.__name__)
        full = get_config()
        self.config = full.get("openrouter_api", {})
        self._fusion_enabled = full.get("fusion", {}).get("enabled", True)
        self.global_response_module = global_response_module
        self._command_functions = {
            "read": self._read,
        }
        self._commands = self._prepare_commands(get_commands_config())
        if fusion is not None:
            self.fusion = fusion
        elif self._fusion_enabled:
            self.fusion = FusionModule(global_response_module)
        else:
            self.fusion = None

    def _prepare_commands(self, raw):
        """Keep bound functionalities in file order. Unbound names do not intercept."""
        if not isinstance(raw, dict):
            self.logger.error(
                "commands_config.yaml must be a mapping of functionality names to rule lists."
            )
            return {}

        prepared = {}
        for name, rules in raw.items():
            handler_name = name if isinstance(name, str) else str(name)
            if handler_name not in self._command_functions:
                self.logger.warning(f"No function bound for command '{handler_name}'.")
                continue
            if not isinstance(rules, list):
                self.logger.warning(f"Command '{handler_name}' rules must be a list.")
                continue
            kept = []
            for rule in rules:
                if not isinstance(rule, str) or not rule.strip():
                    self.logger.warning(
                        f"Command '{handler_name}' has a rule that is not a non-empty string."
                    )
                    continue
                text = rule.strip()
                try:
                    re.compile(text)
                except re.error:
                    self.logger.warning(
                        f"Command '{handler_name}' rule {text!r} is not a valid regex "
                        "and will be used as a keyword."
                    )
                kept.append(text)
            prepared[handler_name] = kept
        return prepared

    def _match_command(self, text_input: str):
        """Return the bound function for the first matching functionality, or None."""
        query = text_input if isinstance(text_input, str) else ""
        for name, rules in self._commands.items():
            for rule in rules:
                if _rule_matches(rule, query):
                    self.logger.info(f"Command '{name}' matched.")
                    return self._command_functions[name]
        return None

    def ProcessRequest(self, text_input: str):
        """
        Answer an active request. A command rule from commands_config.yaml runs
        its bound function and skips the models. Otherwise fusion is the default:
        the local and cloud models run together and speech starts from the local
        stream. With fusion disabled, the cloud model list is tried once and the
        full answer is spoken as a single message.
        """
        handler = self._match_command(text_input)
        if handler is not None:
            return handler(text_input)
        if self.fusion is not None and self._fusion_enabled:
            return self.fusion.handle(text_input)
        return self._cloud_only(text_input)

    def _current_frame(self):
        from modules.shared_buffer import video_buffer
        return video_buffer.retrieve_frame()

    def _run_ocr(self, frame):
        from modules.ai_manager import AI_manager
        return AI_manager.execute_module(
            lambda module: module.__class__.__name__ == "OCRModule",
            "read",
            frame,
        )

    def _read(self, text_input: str) -> str:
        """Run full OCR on the current frame, speak it, and skip the models."""
        self.logger.info(f"Read command: {text_input}")
        result = self._run_ocr(self._current_frame())
        if not result:
            result = READ_ERROR_TEXT
        self.global_response_module.add_message(result, priority=10, message_expiration=30.0)
        return result

    def _cloud_only(self, text_input: str):
        from .ai_manager import AI_manager

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
