# Memory Analyze

An optional Python package that adds truthfulness and importance analysis to `memory_core` using large language models (LLMs). This package integrates seamlessly with `memory_core` to enhance memory processing by evaluating content with configurable LLM backends.

## Features
- **Content Analysis**: Extracts factual statements from text and assigns truthfulness (0-1) and importance (0-1) scores.
- **LLM Support**: Compatible with Hugging Face, Google Gemini, and Groq models.
- **Modular Design**: Works as an optional add-on to `memory_core`, installed separately.

## Installation

Install `memory_analyze` from GitHub:

```bash
pip install git+https://github.com/username/memory_analyze.git
```
### Dependencies
- `huggingface_hub` (for Hugging Face models)
- `google-generativeai` (for Gemini models)
- `groq` (for Groq models)

Install only the dependencies you need based on your chosen LLM provider. For example:
- Hugging Face: `pip install huggingface_hub`
- Gemini: `pip install google-generativeai`
- Groq: `pip install groq`

**Note**: You must have `memory_core` installed to use this package effectively:
```bash
pip install git+https://github.com/username/memory_core.git
```
## Usage

### With `memory_core`
When `memory_analyze` is installed, `memory_core` automatically uses it for content analysis. Pass `analyze_kwargs` to configure the LLM:

```python
from memory_core import MemoryManager, SQLiteBackend

sqlite_backend = SQLiteBackend("memory.db")
memory = MemoryManager(
    long_term_backend=sqlite_backend,
    analyze_kwargs={"model_type": "hf", "api_key": "your_hf_token"}
)
memory.process_content("user", "The sky is blue", "What color is the sky?")
short_term = memory.get_short_term()
print("Short-term:", short_term["documents"], short_term["metadatas"])
```
### Standalone Usage
You can use `analyze_data` directly if desired:

```python
from memory_analyze import analyze_data

facts = analyze_data(
    source="user",
    content="The sky is blue",
    query="What color is the sky?",
    model_type="hf",
    api_key="your_hf_token"
)
print(facts)  # List of dicts with text, truthfulness, and importance
```
### Custom Model Client
Pass a pre-configured model client for more control:

```python
from memory_analyze import analyze_data
from groq import Groq

groq_client = Groq(api_key="your_groq_api_key")
facts = analyze_data(
    source="user",
    content="The sky is blue",
    query="What color is the sky?",
    model_client=groq_client,
    model_type="groq"
)
print(facts)
```
## Configuration
- `model_type`: One of `"hf"` (Hugging Face), `"gemini"`, or `"groq"` (default: `"hf"`).
- `api_key`: Required unless a `model_client` is provided.
- `model_client`: Optional pre-configured client instance (e.g., `InferenceClient`, `GenerativeModel`, `Groq`).

## Supported Models
- **Hugging Face**: Uses `mistralai/Mixtral-8x7B-Instruct-v0.1` by default.
- **Google Gemini**: Uses `gemini-1.5-flash`.
- **Groq**: Uses the default Groq API chat completion.

## Requirements
- An API key for your chosen LLM provider.
- Internet access for model inference.

## Contributing
Submit issues or pull requests to the [GitHub repository](https://github.com/username/memory_analyze).

## License
