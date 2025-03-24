import logging
from huggingface_hub import InferenceClient
import google.generativeai as genai
from groq import Groq

def analyze_data(source, content, query="", model_client=None, model_type="hf", api_key=None):
    """Analyze content to extract facts with truthfulness and importance scores."""
    logger = logging.getLogger(__name__)
    logger.debug(f"Analyzing data: source={source}, query={query}")

    # Initialize model client if not provided
    if not model_client:
        if model_type == "hf":
            model_client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=api_key)
        elif model_type == "gemini":
            genai.configure(api_key=api_key)
            model_client = genai.GenerativeModel("gemini-1.5-flash")
        elif model_type == "groq":
            model_client = Groq(api_key=api_key)
        else:
            raise ValueError("Unsupported model_type or no model_client provided")

    prompt = f"Source: {source}\nContent: {content}\nQuery: {query}\nExtract key factual statements with estimated truthfulness (0-1) and importance (0-1). Return as a list of JSON objects like: {{'text': 'fact', 'truthfulness': 0.7, 'importance': 0.8}}"
    try:
        if model_type == "hf":
            response = model_client.text_generation(prompt, max_new_tokens=4000, temperature=0.7)
        elif model_type == "gemini":
            response = model_client.generate_content(prompt).text
        elif model_type == "groq":
            response = model_client.create_chat_completion(messages=[{"role": "user", "content": prompt}]).choices[0].message.content
        facts = eval(response) if response.startswith("[") else [{"text": response.strip(), "truthfulness": 0.5, "importance": 0.5}]
    except Exception as e:
        logger.error(f"Error in analyze_data: {e}")
        facts = [{"text": f"Failed to analyze: {str(e)}", "truthfulness": 0.1, "importance": 0.1}]
    return facts
