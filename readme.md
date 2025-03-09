
---

```markdown
# 🚀 RealTTS: Real-Time Multimodal Translator

RealTTS is a real-time multimodal translator that converts speech to speech in different languages while also generating images based on the translated text. It leverages cutting-edge AI models like Whisper for ASR (speech recognition), SeamlessM4T for multilingual translation, Stable Diffusion for text-to-image generation, and gTTS for text-to-speech synthesis.

---

## Features

1. **Speech Recognition**: Converts speech to text using OpenAI Whisper.
2. **Multilingual Translation**: Translates recognized text into multiple target languages using SeamlessM4T.
3. **Text-to-Speech Synthesis**: Converts translated text into speech using gTTS.
4. **Image Generation**: Generates images based on translated text prompts using Stable Diffusion.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (optional for faster image generation)

### Steps

1. Clone the repository:
   ```
   git clone 
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

---

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. Open the Gradio interface in your browser.

3. Upload or record an audio file and select source and target languages.

4. View results:
   - Recognized text.
   - Translated text.
   - Translated speech audio.
   - Generated image based on translated text.

---

## Extending to Real-Time Translation Using OpenCV

To make this project real-time using OpenCV:

### 1. Capture Audio in Real-Time
Use OpenCV's `cv2.VideoCapture` or an audio library like `pyaudio` to capture microphone input in real-time.

#### Example Code Snippet for Capturing Audio:
```
import pyaudio

def capture_audio():
    CHUNK = 1024  # Number of audio samples per frame
    FORMAT = pyaudio.paInt16  # Format of audio input
    CHANNELS = 1  # Mono audio
    RATE = 44100  # Sampling rate

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []
    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        print("Stopping recording...")
        stream.stop_stream()
        stream.close()
        p.terminate()

capture_audio()
```

---

### 2. Process Frames and Audio in Real-Time
Feed the captured audio into Whisper for real-time ASR (speech-to-text) processing.

#### Example Integration with Whisper:
```
def process_real_time_audio(audio_chunk):
    # Use Whisper to transcribe audio chunk in real-time
    result = whisper_model.transcribe(audio_chunk)
    return result["text"]
```

---

### 3. Generate Real-Time Outputs
Stream real-time translations and generated images back to the user interface using OpenCV or Gradio's live streaming capabilities.

#### Example Code Snippet for Streaming Text/Images:
```
import cv2

def display_text_on_video(text):
    cap = cv2.VideoCapture(0)  # Open webcam feed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Display translated text on video feed
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Real-Time Translation", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

display_text_on_video("Hello World!")
```

---

## Dependencies

The following libraries are used in this project:

- `torch`: PyTorch framework for deep learning models.
- `transformers`: Hugging Face library for SeamlessM4T and other NLP models.
- `diffusers`: Hugging Face library for Stable Diffusion image generation.
- `openai-whisper`: OpenAI's Whisper model for ASR.
- `gtts`: Google Text-to-Speech library for online TTS.
- `gradio`: Framework for building interactive web interfaces.
- `pyaudio` (optional): For capturing real-time audio input.

Install all dependencies via `requirements.txt`:
```
pip install -r requirements.txt
```

---

## Future Enhancements

1. **Real-Time Streaming**: Implement continuous audio capture and processing with live translation output.
2. **Offline TTS**: Add offline TTS support using libraries like pyttsx3 or Coqui TTS.
3. **Improved Image Generation**: Enhance Stable Diffusion prompts with additional context from ASR results.

---


```

---

