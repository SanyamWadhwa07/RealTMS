import os
import shutil
import torch
from transformers import AutoProcessor, SeamlessM4TModel
from diffusers import StableDiffusionPipeline
from gtts import gTTS
import gradio as gr
import whisper
import tempfile


BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "cache")
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE, "huggingface")
os.environ["WHISPER_CACHE"] = os.path.join(CACHE, "whisper")
os.environ["GRADIO_TEMP_DIR"] = os.path.join(CACHE, "gradio_temp")

# Load models
print("Loading models...")
whisper_model = whisper.load_model("small", download_root=os.environ["WHISPER_CACHE"])
processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
translator = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")
image_gen = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to("cuda" if torch.cuda.is_available() else "cpu")

# Supported languages with their codes for TTS
LANGS = ["English", "Spanish", "French", "German", "Japanese"]
LANG_CODES = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "japanese": "ja"
}

# Completely rewritten audio processing function
def process_audio(audio_input):
    """
    Process audio input from Gradio, handling both file path strings and
    tuple inputs (sample_rate, audio_data).
    """
    print(f"Processing audio input: {type(audio_input)}")
    
    if audio_input is None:
        raise ValueError("No audio input provided")
    
    # Create a stable output location
    final_audio_path = os.path.join(OUT, "input_audio.wav")
    
    # Handle tuple input (sample_rate, audio_data)
    if isinstance(audio_input, tuple) and len(audio_input) == 2:
        sample_rate, audio_data = audio_input
        print(f"Received audio tuple with sample rate {sample_rate}")
        
        # Save the audio data to a WAV file
        try:
            import scipy.io.wavfile as wavfile
            wavfile.write(final_audio_path, sample_rate, audio_data)
            print(f"Saved audio data to {final_audio_path}")
            return final_audio_path
        except Exception as e:
            print(f"Error saving audio data: {str(e)}")
            raise
    
    # Handle file path input (string)
    elif isinstance(audio_input, str):
        print(f"Received audio file path: {audio_input}")
        
        # Check if the file exists
        if not os.path.exists(audio_input):
            print(f"Warning: Audio file not found at {audio_input}")
            raise FileNotFoundError(f"Audio file not found: {audio_input}")
        
        # Copy the file to our stable location
        try:
            shutil.copy2(audio_input, final_audio_path)
            print(f"Copied audio file to {final_audio_path}")
            return final_audio_path
        except Exception as e:
            print(f"Error copying audio file: {str(e)}")
            raise
    
    # Handle unexpected input type
    else:
        input_type = type(audio_input).__name__
        print(f"Unexpected audio input type: {input_type}")
        raise TypeError(f"Unexpected audio input type: {input_type}")

# Speech Recognition 
def transcribe(audio_path):
    try:
        print(f"Transcribing audio from {audio_path}")
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        raise

# Translation Function 
def translate(text, src="English", tgt="French"):
    src_lang = src.lower()
    tgt_lang = tgt.lower()
    try:
        print(f"Translating from {src_lang} to {tgt_lang}: {text[:50]}...")
        inputs = processor(text=text, src_lang=src_lang, return_tensors="pt")
        tokens = translator.generate(**inputs, tgt_lang=tgt_lang)
        return processor.decode(tokens[0].tolist(), skip_special_tokens=True)
    except Exception as e:
        print(f"Translation error: {str(e)}")
        raise

# Text-to-Speech Synthesis
def synthesize(text, lang='fr'):
    lang_code = LANG_CODES.get(lang, 'en')
    audio_out = os.path.join(OUT, "translated_audio.mp3")
    try:
        print(f"Synthesizing speech in {lang_code}: {text[:50]}...")
        tts = gTTS(text=text, lang=lang_code)
        tts.save(audio_out)
        return audio_out
    except Exception as e:
        print(f"TTS error: {str(e)}")
        raise

# Image Generation 
def gen_image(prompt):
    img_out = os.path.join(OUT, "generated_image.png")
    try:
        print(f"Generating image for prompt: {prompt[:50]}...")
        image = image_gen(prompt).images[0]
        image.save(img_out)
        return img_out
    except Exception as e:
        print(f"Image generation error: {str(e)}")
        raise

# Main pipeline function with better error handling and debugging
def pipeline(audio_input, src_lang="English", tgt_lang="French"):
    try:
        if audio_input is None:
            return "Error: No audio input provided", "", None, None
            
        print(f"Pipeline started with audio input type: {type(audio_input)}")
        audio_path = process_audio(audio_input)
        
        text = transcribe(audio_path)
        print(f"Transcription result: {text}")
        
        translated_text = translate(text, src=src_lang, tgt=tgt_lang)
        print(f"Translation result: {translated_text}")
        
        audio_out = synthesize(translated_text, lang=tgt_lang.lower())
        
        img_out = gen_image(translated_text)
        
        return text, translated_text, audio_out, img_out
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Pipeline error: {error_msg}")
        import traceback
        traceback.print_exc()
        return error_msg, "", None, None

# Gradio interface with correct audio component configuration
with gr.Blocks() as demo:
    
    gr.Markdown("# 🚀 RealTTS: Real-Time Multimodal Translator")

    with gr.Row():
        src_dropdown = gr.Dropdown(choices=LANGS,
                                   label="Source Language",
                                   value="English")
        
        tgt_dropdown = gr.Dropdown(choices=LANGS,
                                   label="Target Language",
                                   value="French")

    # Use mic and upload options but change audio format
    audio_input = gr.Audio(
        label="🎙️ Upload or Record Audio",
        sources=["microphone", "upload"],
        type="numpy"  # Use numpy array format instead of filepath
    )
    
    submit_btn = gr.Button("Translate 🚀")

    with gr.Row():
        text_outbox = gr.Textbox(label="🎤 Recognized Text")
        translated_outbox = gr.Textbox(label="🌐 Translated Text")

    with gr.Row():
        audio_outbox = gr.Audio(label="🔊 Translated Audio")
        image_outbox = gr.Image(label="🖼️ Generated Image")

    submit_btn.click(
        fn=pipeline,
        inputs=[audio_input, src_dropdown, tgt_dropdown],
        outputs=[text_outbox,
                 translated_outbox,
                 audio_outbox,
                 image_outbox],
    )
    
    # Add a clear button for better UX
    clear_btn = gr.Button("Clear")
    clear_btn.click(
        lambda: [None, None, None, None],
        inputs=None,
        outputs=[text_outbox, translated_outbox, audio_outbox, image_outbox]
    )

if __name__ == "__main__":
    # Add more verbose error output
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Start the interface with debug mode enabled
    demo.launch(debug=True)