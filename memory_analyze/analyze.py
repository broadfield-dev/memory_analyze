import logging
import os
from huggingface_hub import InferenceClient
import google.generativeai as genai
from groq import Groq

def analyze_data(source, content, query='', model_client=None, model_type='hf', api_key=None):
    '''Analyze content to extract facts with truthfulness and importance scores.'''
    logger = logging.getLogger(__name__)
    logger.debug(f'Analyzing data: source={source}, query={query}')

    # Pull API keys from environment if not provided
    if not api_key:
        if model_type == 'hf':
            api_key = os.environ.get('HF_TOKEN')
        elif model_type == 'gemini':
            api_key = os.environ.get('GOOGLE_API_KEY')
        elif model_type == 'groq':
            api_key = os.environ.get('GROQ_API_KEY')

    # Initialize model client if not provided
    if not model_client:
        if not api_key:
            logger.warning(f'No API key found for {model_type} in environment or arguments. Returning defaults.')
            return [{'text': content, 'truthfulness': 0.5, 'importance': 0.5}]
        
        try:
            if model_type == 'hf':
                model_client = InferenceClient(model='mistralai/Mixtral-8x7B-Instruct-v0.1', token=api_key)
            elif model_type == 'gemini':
                genai.configure(api_key=api_key)
                model_client = genai.GenerativeModel('gemini-2.0-flash')
            elif model_type == 'groq':
                model_client = Groq(api_key=api_key)
            else:
                raise ValueError('Unsupported model_type')
        except Exception as e:
            logger.error(f'Failed to initialize {model_type} client: {e}')
            return [{'text': f'Failed to initialize model: {str(e)}', 'truthfulness': 0.1, 'importance': 0.1}]

    prompt = f'Source: {source}\nContent: {content}\nQuery: {query}\nExtract key factual statements with estimated truthfulness (0-1) and importance (0-1). Return as a list of JSON objects like: {{\'text\': \'fact\', \'truthfulness\': 0.7, \'importance\': 0.8}}'
    try:
        if model_type == 'hf':
            response = model_client.text_generation(prompt, max_new_tokens=4000, temperature=0.7)
        elif model_type == 'gemini':
            response = model_client.generate_content(prompt).text
        elif model_type == 'groq':
            response = model_client.create_chat_completion(messages=[{'role': 'user', 'content': prompt}]).choices[0].message.content
        facts = eval(response) if response.startswith('[') else [{'text': response.strip(), 'truthfulness': 0.5, 'importance': 0.5}]
    except Exception as e:
        logger.error(f'Error in analyze_data: {e}')
        facts = [{'text': f'Failed to analyze: {str(e)}', 'truthfulness': 0.1, 'importance': 0.1}]
    return facts
