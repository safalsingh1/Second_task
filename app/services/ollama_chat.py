from typing import Any, Dict, List, Type
import json
from pydantic import BaseModel
from langchain_community.llms import Ollama

class OllamaChat:
    def __init__(self, model_name: str = "llama3.1"):
        self.client = Ollama(model=model_name)

    def chat_completion_create(self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs) -> Any:
        # Convert chat messages to a single prompt
        prompt = self._convert_messages_to_prompt(messages)
        
        # Get completion from Ollama
        response = self.client.invoke(
            f"""
            Please provide your response in the following JSON format:
            {{
                "thought_process": ["thought 1", "thought 2", ...],
                "answer": "your detailed answer here",
                "enough_context": true or false
            }}

            User Query: {prompt}
            """
        )
        
        # Try to extract JSON from the response
        try:
            # Find the JSON part in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return response_model.model_validate(json.loads(json_str))
            else:
                # If no JSON found, create a structured response
                return response_model(
                    thought_process=["Processed the query", "Generated plain text response"],
                    answer=response,
                    enough_context=True
                )
        except json.JSONDecodeError:
            # Handle non-JSON responses
            return response_model(
                thought_process=["Processed the query", "Generated plain text response"],
                answer=response,
                enough_context=True
            )

    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a text prompt."""
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        return prompt.strip()
