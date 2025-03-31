import logging
import os
import json
import re
from huggingface_hub import InferenceClient
import google.generativeai as genai
from groq import Groq

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def call_gemini(app_config, model_name, prompt):
    """Calls the Google Gemini model and ensures raw output with proper spacing."""
    api_key = app_config.get('GOOGLE_GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_GEMINI_API_KEY is not set!")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    try:
        response = model.generate_content(prompt, stream=True)

        def strip_generator(response):
            buffer = ""
            in_body = False
            last_char = None
            for chunk in response:
                text = chunk.text if hasattr(chunk, 'text') else ''
                if last_char and text:
                    if (last_char.isalnum() or last_char in '>-') and (text[0].isalnum() or text[0] in '<'):
                        buffer += " "
                    elif last_char == ' ' and text[0] == ' ':
                        text = text.lstrip()
                buffer += text
                last_char = text[-1] if text else None

                if not in_body:
                    body_start = buffer.find('<body>')
                    if body_start != -1:
                        buffer = buffer[body_start + 6:]
                        in_body = True
                    elif '<h1>' in buffer or '<p>' in buffer:
                        in_body = True

                if in_body:
                    buffer = re.sub(r'^```html\s*|\s*```$', '', buffer, flags=re.MULTILINE)
                    buffer = re.sub(r'^<code>\s*|\s*</code>$', '', buffer)
                    buffer = re.sub(r'^["\']|["\']$', '', buffer)
                    buffer = buffer.replace('\\"', '"').replace("\\'", "'")

                    body_end = buffer.find('</body>')
                    if body_end != -1:
                        chunk_text = buffer[:body_end]
                        buffer = buffer[body_end + 7:]
                        if chunk_text.strip():
                            yield chunk_text.strip()
                    elif len(buffer) > 0:
                        chunk_text = buffer
                        buffer = ""
                        if chunk_text.strip():
                            yield chunk_text.strip()

            if buffer.strip():
                buffer = re.sub(r'^```html\s*|\s*```$', '', buffer, flags=re.MULTILINE)
                buffer = re.sub(r'^<code>\s*|\s*</code>$', '', buffer)
                buffer = re.sub(r'^["\']|["\']$', '', buffer)
                buffer = buffer.replace('\\"', '"').replace("\\'", "'")
                yield buffer.strip()

        return strip_generator(response)
    except Exception as e:
        raise Exception(f"Gemini API Error: {e}")

def call_groq(app_config, model_name, prompt):
    """Calls the Groq model."""
    api_key = app_config.get('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set!")

    client = Groq(api_key=api_key)
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            stream=True
        )
        return chat_completion
    except Exception as e:
        raise Exception(f"Groq API Error: {e}")

def call_huggingface(app_config, model_name, prompt):
    """Calls the Hugging Face model."""
    api_key = app_config.get('HF_TOKEN')
    if not api_key:
        raise ValueError("HF_TOKEN is not set!")

    client = InferenceClient(model=model_name, token=api_key)
    try:
        response = client.text_generation(prompt, max_new_tokens=4000, temperature=0.7, stream=True)
        return response  # Returns a generator of chunks
    except Exception as e:
        raise Exception(f"Hugging Face API Error: {e}")

def call_xai(app_config, model_name, prompt):
    """Placeholder for calling the xAI model (replace with actual API if available)."""
    api_key = app_config.get('XAI_API_KEY')
    if not api_key:
        raise ValueError("XAI_API_KEY is not set!")

    # Mock implementation (replace with real xAI API call)
    logger.warning("xAI API not implemented; returning mock response")
    try:
        from groq import Groq  # Using Groq as a placeholder for xAI
        client = Groq(api_key=api_key)  # Replace with actual xAI client
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            stream=True
        )
        return chat_completion
    except Exception as e:
        raise Exception(f"xAI API Error: {e}")

def analyze_data(source, content, query='', model_client=None, model_type='gemini', api_key=None):
    """Analyze content to extract facts with truthfulness, importance, and sentiment scores."""
    logger.debug(f'Analyzing data: source={source}, content={content}, query={query}, model_type={model_type}')

    # App config with API keys from environment if not provided
    app_config = {
        'HF_TOKEN': os.environ.get('HF_TOKEN', ''),
        'GOOGLE_GEMINI_API_KEY': os.environ.get('GOOGLE_GEMINI_API_KEY', ''),
        'GROQ_API_KEY': os.environ.get('GROQ_API_KEY', ''),
        'XAI_API_KEY': os.environ.get('XAI_API_KEY', '')
    }

    # Model names
    model_names = {
        'hf': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'gemini': 'gemini-1.5-flash',
        'groq': 'mixtral-8x7b-32768',  # Example Groq model
        'xai': 'grok'  # Hypothetical xAI model name
    }

    prompt = (
        f'Source: {source}\nContent: {content}\nQuery: {query}\n'
        f'Extract key factual statements with estimated truthfulness (0-1), importance (0-1), '
        f'and sentiment (positive, negative, neutral). Return as a list of JSON objects like: '
        f'{{\'text\': \'fact\', \'truthfulness\': 0.7, \'importance\': 0.8, \'sentiment\': \'positive\'}}'
    )

    try:
        if model_type == 'hf':
            response = call_huggingface(app_config, model_names['hf'], prompt)
        elif model_type == 'gemini':
            response = call_gemini(app_config, model_names['gemini'], prompt)
        elif model_type == 'groq':
            response = call_groq(app_config, model_names['groq'], prompt)
        elif model_type == 'xai':
            response = call_xai(app_config, model_names['xai'], prompt)
        else:
            raise ValueError(f'Unsupported model_type: {model_type}')

        # Process streaming response into a single string
        full_response = ''
        for chunk in response:
            if model_type == 'groq' or model_type == 'xai':
                full_response += chunk.choices[0].delta.content or ''  # Groq/xAI streaming format
            else:
                full_response += chunk  # HF/Gemini format

        logger.debug(f'Raw LLM response: {full_response}')

        # Parse the response
        try:
            full_response = full_response.strip()
            if full_response.startswith('[') and full_response.endswith(']'):
                facts = json.loads(full_response)
            else:
                logger.warning(f'Response not a JSON list: {full_response}')
                facts = [{'text': content, 'truthfulness': 0.5, 'importance': 0.5, 'sentiment': 'neutral'}]
        except json.JSONDecodeError as e:
            logger.error(f'JSON parsing failed: {e}, response={full_response}')
            facts = [{'text': content, 'truthfulness': 0.5, 'importance': 0.5, 'sentiment': 'neutral'}]

        # Validate and normalize facts
        valid_facts = []
        valid_sentiments = {'positive', 'negative', 'neutral'}
        for fact in facts:
            if not isinstance(fact, dict) or not all(k in fact for k in ['text', 'truthfulness', 'importance', 'sentiment']):
                logger.warning(f'Invalid fact format: {fact}')
                valid_facts.append({'text': fact.get('text', content), 'truthfulness': 0.5, 'importance': 0.5, 'sentiment': 'neutral'})
            else:
                try:
                    fact['truthfulness'] = max(0.0, min(1.0, float(fact['truthfulness'])))
                    fact['importance'] = max(0.0, min(1.0, float(fact['importance'])))
                    sentiment = fact.get('sentiment', 'neutral').lower()
                    fact['sentiment'] = sentiment if sentiment in valid_sentiments else 'neutral'
                    valid_facts.append(fact)
                except (ValueError, TypeError) as e:
                    logger.error(f'Invalid numeric value in fact: {fact}, error: {e}')
                    valid_facts.append({'text': fact['text'], 'truthfulness': 0.5, 'importance': 0.5, 'sentiment': 'neutral'})

        logger.info(f'Processed facts: {valid_facts}')
        return valid_facts

    except Exception as e:
        logger.error(f'Error during analysis: {e}')
        return [{'text': f'Analysis failed: {str(e)}', 'truthfulness': 0.1, 'importance': 0.1, 'sentiment': 'neutral'}]
