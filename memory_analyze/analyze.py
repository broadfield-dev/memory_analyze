import logging
import os
from huggingface_hub import InferenceClient
import google.generativeai as genai
from groq import Groq
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def analyze_data(source, content, query='', model_client=None, model_type='hf', api_key=None):
    '''Analyze content to extract facts with truthfulness and importance scores.'''
    logger.debug(f'Analyzing data: source={source}, content={content}, query={query}, model_type={model_type}')

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
                model_client = genai.GenerativeModel('gemini-1.5-flash')
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
        
        logger.debug(f'Raw LLM response: {response}')
        
        # Robust parsing of response
        try:
            # Strip any leading/trailing whitespace or non-JSON content
            response = response.strip()
            if response.startswith('[') and response.endswith(']'):
                facts = json.loads(response)  # Use json.loads instead of eval for safety
            else:
                # Handle case where response isn’t a list
                facts = [{'text': response.strip(), 'truthfulness': 0.5, 'importance': 0.5}]
        except json.JSONDecodeError as e:
            logger.error(f'Failed to parse LLM response as JSON: {e}, response={response}')
            facts = [{'text': response.strip(), 'truthfulness': 0.5, 'importance': 0.5}]
        
        # Validate facts format
        for fact in facts:
            if not all(k in fact for k in ['text', 'truthfulness', 'importance']):
                logger.warning(f'Invalid fact format: {fact}, using defaults')
                fact.update({'text': fact.get('text', content), 'truthfulness': 0.5, 'importance': 0.5})
            # Ensure values are within 0-1 range
            fact['truthfulness'] = max(0.0, min(1.0, float(fact['truthfulness'])))
            fact['importance'] = max(0.0, min(1.0, float(fact['importance'])))

    except Exception as e:
        logger.error(f'Error in analyze_data: {e}')
        facts = [{'text': f'Failed to analyze: {str(e)}', 'truthfulness': 0.1, 'importance': 0.1}]
    
    logger.debug(f'Final facts: {facts}')
    return facts
