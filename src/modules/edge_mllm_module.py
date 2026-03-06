from ollama import chat
from .ai_module_base import BaseAIModel

class EdgeMLLMModule(BaseAIModel):
    """
    Module responsible for generating text and interacting with local Ollama models.
    Replaces the functionality of junk/ModelManager.py.
    """
    def __init__(self):
        super().__init__("ollama_llm")
        self.model_name = self.config.get("model", "gemma3:4b")

    def load_model(self):
        """Loads the local Ollama LLM into memory."""
        self.logger.info(f"Loading Edge MLLM model ({self.model_name})...")
        try:
            # Trigger a small load to ensure model is in VRAM if needed
            chat(model=self.model_name, messages=[{'role':'user','content':'ping'}])
            self.model = self.model_name # Assign simply to denote loaded state
            self.logger.info("Local LLM Loaded successfully.")
        except Exception as e:
            self.logger.error(f"Could not load Local LLM. Ensure Ollama is running. Error: {e}")
            raise

    def run_inference(self, input_data: str, **kwargs):
        """
        Generator function that streams text from Ollama.
        
        Args:
            input_data (str): The prompt text to send to the model.
            kwargs:
                image_path (str): Optional. If provided, uses the image for vision generation.
                already_spoken (str): Optional. If provided, provides conversational context assistant context.
        
        Yields:
            str: Chunks of the generated response.
        """
        prompt = input_data
        image_path = kwargs.get("image_path")
        already_spoken = kwargs.get("already_spoken")

        try:
            # Construct messages based on provided arguments
            if image_path:
                complete_prompt = f"Concisely answer this query in paragraph form using the image. {prompt}"
                messages = [
                    {
                        'role': 'user',
                        'content': complete_prompt,
                        'images': [image_path]
                    }
                ]
            elif already_spoken is not None:
                messages = [
                    {'role': 'user', 'content': prompt}, 
                    {'role': 'assistant', 'content': already_spoken}
                ]
            else:
                messages = [
                    {'role': 'user', 'content': prompt}
                ]

            # Enable streaming
            stream = chat(
                model=self.model_name,
                messages=messages,
                stream=True
            )

            for chunk in stream:
                # Ollama yields objects with 'message' -> 'content'
                content = chunk.get('message', {}).get('content', '')
                if content:
                    content = content.replace("*", "")
                    yield content

        except Exception as e:
            self.logger.error(f"Ollama Error: {e}")
            yield f" [Local Error: {str(e)}] "
