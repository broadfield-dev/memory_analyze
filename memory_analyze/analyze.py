import logging
import os
from huggingface_hub import InferenceClient
import google.generativeai as genai
from groq import Groq
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_data(source, content, query='', model_client=None, model_type='hf', api_key=None):
    '''Analyze content to extract facts with truthfulness and importance scores.'''
    logger.debug(f'Entering analyze_data: source={source}, content={content}, query={query}, model_type={model_type}, api_key_provided={bool(api_key)}')

    # Pull API keys from environment if not provided
    if not api_key:
        if model_type == 'hf':
            api_key = os.environ.get('HF_TOKEN')
        elif model_type == 'gemini':
            api_key = os.environ.get('GOOGLE_API_KEY')
        elif model_type == 'groq':
            api_key = os.environ.get('GROQ_API_KEY')
        logger.debug(f'API key from env: {api_key}')

    # Initialize model client if not provided
    if not model_client:
        if not api_key:
            logger.warning(f'No API key for {model_type}; returning defaults')
            return [{'text': content, 'truthfulness': 0.5, 'importance': 0.5}]
        
        try:
            if model_type == 'hf':
                model_client = InferenceClient(model='mistralai/Mixtral-8x7B-Instruct-v0.1', token=api_key)
                logger.info('Hugging Face client initialized')
            elif model_type == 'gemini':
                genai.configure(api_key=api_key)
                model_client = genai.GenerativeModel('gemini-1.5-flash')
                logger.info('Gemini client initialized')
            elif model_type == 'groq':
                model_client = Groq(api_key=api_key)
                logger.info('Groq client initialized')
            else:
                logger.error(f'Unsupported model_type: {model_type}')
                return [{'text': f'Unsupported model: {model_type}', 'truthfulness': 0.1, 'importance': 0.1}]
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
        
        logger.debug(f'LLM raw response: {response}')
        
        # Robust parsing
        try:
            response = response.strip()
            if response.startswith('[') and response.endswith(']'):
                facts = json.loads(response)
            else:
                logger.warning(f'Response not a JSON list: {response}')
                facts = [{'text': response.strip(), 'truthfulness': 0.5, 'importance': 0.5}]
        except json.JSONDecodeError as e:
            logger.error(f'JSON parsing failed: {e}, response={response}')
            facts = [{'text': response.strip(), 'truthfulness': 0.5, 'importance': 0.5}]

        # Validate and normalize facts
        valid_facts = []
        for fact in facts:
            if not isinstance(fact, dict) or not all(k in fact for k in ['text', 'truthfulness', 'importance']):
                logger.warning(f'Invalid fact format: {fact}')
                valid_facts.append({'text': fact.get('text', content), 'truthfulness': 0.5, 'importance': 0.5})
            else:
                try:
                    fact['truthfulness'] = max(0.0, min(1.0, float(fact['truthfulness'])))
                    fact['importance'] = max(0.0, min(1.0, float(fact['importance'])))
                    valid_facts.append(fact)
                except (ValueError, TypeError) as e:
                    logger.error(f'Invalid numeric value in fact: {fact}, error: {e}')
                    valid_facts.append({'text': fact['text'], 'truthfulness': 0.5, 'importance': 0.5})

        logger.info(f'Processed facts: {valid_facts}')
        return valid_facts

    except Exception as e:
        logger.error(f'Error during analysis: {e}')
        return [{'text': f'Analysis failed: {str(e)}', 'truthfulness': 0.1, 'importance': 0.1}]
