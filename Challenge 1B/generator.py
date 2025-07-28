# generator.py

from ctransformers import AutoModelForCausalLM
from typing import List, Dict

class Generator:
    """
    Uses the official, stable, and quantized Qwen1.5-0.5B-Chat GGUF model.
    """
    def __init__(self, model_repo: str = "Qwen/Qwen3-0.6B-GGUF", model_file="Qwen3-0.6B-Q8_0.gguf"):
        """
        Initializes the Qwen GGUF model using ctransformers.

        Args:
            model_repo: The official Hugging Face repository for the GGUF model.
            model_file: The specific .gguf file to use. The Q4_K_M version has the
                        best balance of size (~407MB) and quality.
        """
        print(f"\nInitializing GGUF Generator with model: {model_repo}/{model_file}...")
        try:
            # Load the quantized Qwen model from the official repository
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_repo,
                model_file=model_file,
                model_type='qwen3',
                local_files_only=False
            )
            print("Official Qwen GGUF Generator initialized successfully.")
        except Exception as e:
            print(f"Error initializing GGUF generator: {e}")
            self.llm = None

    def create_plan_text(self, persona: Dict, job: Dict, context_chunks: List[Dict]) -> str:
        """
        Generates a simple, text-based plan using the Qwen GGUF model.
        """
        if not self.llm:
            return "Error: Generator model not available."

        prompt = self._build_prompt(persona, job, context_chunks)
        
        print("Generating plan text with Qwen GGUF model...")
        response_text = self.llm(
            prompt, 
            max_new_tokens=300, 
            temperature=0.2, 
            top_p=0.95, 
            repetition_penalty=1.1
        )

        return response_text.strip()

    def _build_prompt(self, persona: Dict, job: Dict, context_chunks: List[Dict]) -> str:
        """
        Helper function to construct the prompt using the official ChatML format
        required by the Qwen models for optimal performance.
        """
        system_prompt = (
            f"You are a helpful assistant acting as a {persona.get('role', 'expert')}. "
            f"Your task is to {job.get('task', 'provide a detailed analysis')}. "
            "Your response MUST be based ONLY on the information in the context provided. "
            "Present your answer as a structured summary. Use clear headings for each distinct point. Each heading must end with a colon (:)."
        )
        
        user_prompt = "--- CONTEXT ---\n"
        for chunk in context_chunks:
            first_sentence = chunk['text_content'].split('.')[0]
            user_prompt += f"- Regarding '{chunk['heading_text']}': {first_sentence}.\n"
        
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        return prompt