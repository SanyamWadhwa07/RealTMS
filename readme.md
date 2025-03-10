# 🚀 RealTMS: Real-Time Multimodal Translator

RealTMS is a real-time multimodal translator that converts speech to speech in different languages while also generating images based on the translated text. It leverages cutting-edge AI models like Whisper for ASR (speech recognition), SeamlessM4T for multilingual translation, Stable Diffusion for text-to-image generation, and gTTS for text-to-speech synthesis.

## Features

1. **Speech Recognition**: Converts speech to text using OpenAI Whisper.
2. **Multilingual Translation**: Translates recognized text into multiple target languages using SeamlessM4T.
3. **Text-to-Speech Synthesis**: Converts translated text into speech using gTTS.
4. **Image Generation**: Generates images based on translated text prompts using Stable Diffusion.

## Installation

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (optional for faster image generation)
- **FFmpeg** (required for audio processing)

### Steps

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/RealTTS.git
   cd RealTTS
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Linux/MacOS
   venv\Scripts\activate     # On Windows
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install FFmpeg (required for audio processing):
   - **Windows**: 
     - Download from https://ffmpeg.org/download.html or use a pre-built package
     - Extract to a location (e.g., `C:\ffmpeg`)
     - Add the `bin` folder to your PATH environment variable
     - Verify installation by typing `ffmpeg -version` in a new command prompt
   
   - **macOS**:
     ```
     brew install ffmpeg
     ```
   
   - **Linux**:
     ```
     sudo apt update
     sudo apt install ffmpeg
     ```

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. Open the Gradio interface in your browser.

3. Upload or record an audio file and select source and target languages.

4. View results:
   - Recognized text
   - Translated text
   - Translated speech audio
   - Generated image based on translated text

## Troubleshooting

### Common Issues

1. **FFmpeg Not Found Error**:
   If you encounter an error like:
   ```
   RuntimeWarning: Couldn't find ffprobe or avprobe - defaulting to ffprobe, but may not work
   ```
   or
   ```
   RuntimeError: Cannot load audio from file: `ffprobe` not found. Please install `ffmpeg`
   ```
   
   Follow the FFmpeg installation instructions in the Prerequisites section.

2. **CUDA/GPU Issues**:
   If you encounter CUDA errors, try setting the environment variable to use CPU:
   ```
   export FORCE_CPU=1  # Linux/MacOS
   set FORCE_CPU=1     # Windows
   ```

## Dependencies

The following libraries are used in this project:

- `torch`: PyTorch framework for deep learning models
- `transformers`: Hugging Face library for SeamlessM4T and other NLP models
- `diffusers`: Hugging Face library for Stable Diffusion image generation
- `openai-whisper`: OpenAI's Whisper model for ASR
- `gtts`: Google Text-to-Speech library for online TTS
- `gradio`: Framework for building interactive web interfaces
- `pyaudio` (optional): For capturing real-time audio input
- `ffmpeg`: Required for audio processing (external dependency)

Install all dependencies via `requirements.txt`:
```
pip install -r requirements.txt
```

## Future Enhancements

1. **Real-Time Streaming**: Implement continuous audio capture and processing with live translation output
2. **Offline TTS**: Add offline TTS support using libraries like pyttsx3 or Coqui TTS
3. **Improved Image Generation**: Enhance Stable Diffusion prompts with additional context from ASR results
4. **Automatic FFmpeg Installation**: Add script to automatically install FFmpeg when missing
