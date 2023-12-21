import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import whisper
import scipy




###############################################################
# Initialize the Whisper model
whisper_model = whisper.load_model("large-v3")

###############################################################

# Initialize the bark TTS model
#synthesiser = pipeline("text-to-speech", "suno/bark")

#################

# Initialize the MS TTS model
import soundfile as sf
from datasets import load_dataset
import torch
import numpy as np
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

def divide_text(text, max_length):
    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        # Find the nearest space to the max_length
        split_index = text.rfind(' ', 0, max_length)
        if split_index == -1:  # No space found, forced to split at max_length
            split_index = max_length
        chunks.append(text[:split_index])
        text = text[split_index:].lstrip()  # Remove leading spaces from the next chunk
    return chunks

#

###############################################################


####################################
## Initialize the Mistral-7B model
model_name_or_path = "./LLM/TheBloke_Mistral-7B-Instruct-v0.1-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)




# Speech Recognition Function
def recognize_speech(audio_file):
    print("audio_file:",audio_file)
    text = whisper_model.transcribe(audio_file)['text']
    return text




# Main Function to Process Voice and Generate Response
def process_voice():
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(streaming=True, type="filepath", label="Your Voice")
            recognized_text = gr.Textbox(label="Recognized Text")
            response_voice = gr.Audio(label="AI Response", type="filepath")
            submit_button = gr.Button("Talk to AI")

    def process_and_respond(audio_input):
        # Convert voice to text
        print("audio_input:",audio_input)
        text = recognize_speech(audio_input)
        #recognized_text.update(value=text)
        print("Recognized Text:", text)  # Debug print
        user_message = text



        # Generate response

        # Generate response using LLM
        prompt_template=f'''<s>[INST] {text} [/INST]'''
        response_text = pipe(prompt_template)[0]['generated_text']
        response_text = response_text[len(prompt_template):]
        print("Response Text:", response_text)  # Debug print

        # Convert text response to speech

        #MS TTS
        text_chunks = divide_text(response_text, 600)
        # Process each chunk and collect the audio data
        audio_data = []
        for chunk in text_chunks:
            result = synthesiser(chunk, forward_params={"speaker_embeddings": speaker_embedding})
            audio_data.append(result["audio"])

        # Combine the audio data from all chunks
        combined_audio = np.concatenate(audio_data)

        # Write the combined audio to a file
        sf.write("response.wav", combined_audio, samplerate=result["sampling_rate"])

        #response_voice.update(value="response.wav")
        return text, "response.wav"

    submit_button.click(process_and_respond, inputs=audio_input,outputs=[recognized_text, response_voice])

# Launch the Gradio Blocks interface
with gr.Blocks() as demo:
    process_voice()



demo.launch(share=True)