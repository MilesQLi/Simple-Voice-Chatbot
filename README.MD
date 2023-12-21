# Simple Voice Chatbot

## Overview
This project is an open-source voice chatbot that leverages state-of-the-art AI models to recognize user voice input, generate a textual response using the Mistral-7B-Instruct-v0.1 model, and then convert this response back into speech using a TTS (Text-to-Speech) model.

### Features
- **Voice Recognition**: Using Whisper's large-v3 model for accurate speech-to-text conversion.
- **AI-Powered Text Response**: Utilizes the Mistral-7B-Instruct-v0.1 model for generating human-like text responses.
- **Text-to-Speech**: Converts the AI response into speech using Microsoft's SpeechT5 TTS model.
- **Interactive Interface**: Built with Gradio for an easy-to-use, interactive experience.

## Installation
To run this project, you need to install the required libraries. You can do this by running:
pip install gradio transformers whisper scipy soundfile datasets torch numpy

## Usage
Run the script and interact with the chatbot through the Gradio interface. Speak into your microphone, and the chatbot will respond with voice output.

## How It Works
1. **Voice to Text**: The Whisper model transcribes the user's speech into text.
2. **Generating Response**: The Mistral-7B-Instruct-v0.1 model processes the transcribed text and generates a response.
3. **Text to Speech**: The Microsoft SpeechT5 TTS model converts the generated text response back into speech.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Whisper Model for voice recognition.
- Mistral-7B-Instruct-v0.1 model for text generation.
- Microsoft's SpeechT5 TTS for text-to-speech conversion.
- Gradio for the interactive interface.

## Disclaimer
This project is not officially associated with the creators of the models and tools used in this chatbot. It's an independent project developed and maintained by contributors.
