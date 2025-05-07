import gradio as gr
import torch
import torch.nn.functional as F
import librosa
import numpy as np
from PIL import Image
import os
import joblib
from torchvision import models, transforms
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
)
import datetime
import json
import wave
import contextlib
from typing import Union, Tuple, Dict
import soundfile as sf
from joblib import load
import tempfile
 
# ---------- CONFIG ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# ---------- TEXT MODEL ----------
load_directory = "bert_emotion_model"

# Load the tokenizer and model
text_tokenizer = AutoTokenizer.from_pretrained(load_directory)
text_model = AutoModelForSequenceClassification.from_pretrained(load_directory).to(device)

# Load the label encoder for text classification
label_encoder_text = load(os.path.join(load_directory, "label_encoder.joblib"))

def predict_text(text):
    # Tokenize and prepare input
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    
    with torch.no_grad():
        # Perform prediction
        outputs = text_model(**inputs)
        
        # Log the raw logits
        print("Raw logits:", outputs.logits)
        
        # Apply softmax to get probabilities
        probs = F.softmax(outputs.logits, dim=1)
        
        # Log the probabilities
        print("Probabilities:", probs)
        print("Label Encoder Classes:", label_encoder_text.classes_)

        # Get the predicted index
        pred_idx = torch.argmax(probs, dim=1).item()
        
        # Get the predicted label and confidence
        pred_label = label_encoder_text.inverse_transform([pred_idx])[0]
        confidence = round(probs[0][pred_idx].item(), 4)
    
    return pred_label, confidence

# ---------- VOICE MODEL ----------
voice_model_path = "wav2vec2-emotion-model"
voice_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    voice_model_path,
    use_safetensors=True
).to(device)
voice_processor = Wav2Vec2Processor.from_pretrained(voice_model_path)
voice_model.eval()
id2label_voice = voice_model.config.id2label
 
def predict_voice(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = voice_processor(speech, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        logits = voice_model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_label = id2label_voice[pred_idx]
        confidence = float(round(probs[0][pred_idx].item(), 4))
    return pred_label, confidence
 
# ---------- IMAGE MODEL ----------
class_labels = ['angry', 'fear', 'happy', 'sad', 'surprise']
# Transformation to resize, normalize and convert the image to a tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
 
# ---------- IMAGE MODEL ----------
image_model_path = "final_emotion_image_classifier.pth"  # path to the trained image model
 
# Load the ResNet18 model
image_model = models.resnet18(pretrained=False)
image_model.fc = torch.nn.Linear(image_model.fc.in_features, len(class_labels))
image_model.load_state_dict(torch.load(image_model_path, map_location=device))
image_model = image_model.to(device)
image_model.eval()
 
def predict_image(image: Union[Image.Image, np.ndarray]) -> Tuple[str, float]:
    """
    Function to predict the emotion from the uploaded image.
    Args:
    - image (PIL.Image or np.ndarray): The input image for emotion classification
    Returns:
    - (str, float): Predicted emotion label and confidence score
    """
    # Ensure the image is a PIL image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)  # Convert numpy ndarray to PIL Image
 
    # Apply transformations to the image
    img_tensor = transform(image).unsqueeze(0).to(device)
 
    # Make prediction with no gradient computation
    with torch.no_grad():
        logits = image_model(img_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)  # Get probabilities
        pred_idx = torch.argmax(probs, dim=1).item()  # Get the index of the predicted label
        pred_label = class_labels[pred_idx]  # Get the corresponding label
        confidence = round(probs[0][pred_idx].item(), 4)  # Confidence score
 
    return pred_label, confidence
 
# ---------- FUSION ----------
def fusion_predict(text, audio_path, image):
    results = []
    confidences = []
    
    # Process text if provided
    if text and text.strip():
        try:
            text_label, text_conf = predict_text(text)
            results.append(text_label)
            confidences.append(text_conf)
        except Exception as e:
            print(f"Warning: Text processing failed: {e}")
            text_label, text_conf = "neutral", 0.0
    else:
        text_label, text_conf = "neutral", 0.0
    
    # Process audio if provided
    if audio_path and os.path.exists(audio_path):
        try:
            voice_label, voice_conf = predict_voice(audio_path)
            results.append(voice_label)
            confidences.append(voice_conf)
        except Exception as e:
            print(f"Warning: Audio processing failed: {e}")
            voice_label, voice_conf = "neutral", 0.0
    else:
        voice_label, voice_conf = "neutral", 0.0
    
    # Process image if provided
    if image is not None:
        try:
            image_label, image_conf = predict_image(image)
            results.append(image_label)
            confidences.append(image_conf)
        except Exception as e:
            print(f"Warning: Image processing failed: {e}")
            image_label, image_conf = "neutral", 0.0
    else:
        image_label, image_conf = "neutral", 0.0
    
    # If no inputs were processed, return default values
    if not results:
        return "neutral", 0.0, "neutral", 0.0, "neutral", 0.0, "neutral", 0.0
    
    # Get the most common prediction
    final_label = max(set(results), key=results.count)
    # Calculate average confidence
    final_confidence = round(sum(confidences) / len(confidences), 4)
    
    # Return all predictions with their confidences
    return (
        final_label,
        float(final_confidence),
        text_label if text and text.strip() else "neutral",
        float(text_conf) if text and text.strip() else 0.0,
        voice_label,
        float(voice_conf),
        image_label,
        float(image_conf)
    )

 
# ---------- LOGGING ----------
def log_interaction(user_id, text, image_path, audio_path, result):
    log = {
        "user_id": user_id,
        "timestamp": str(datetime.datetime.now()),
        "text": text,
        "image": image_path,
        "audio": audio_path,
        "result": result
    }
    with open("user_logs.json", "a") as f:
        f.write(json.dumps(log) + "\n")
 
# ---------- MINDFULNESS + PROMPT ----------
def get_mindfulness_suggestions(emotion):
    return {
        "angry": "Take deep breaths and try to release your frustration.",
        "fear": "Acknowledge your fears, but focus on grounding yourself in the present moment.",
        "happy": "Enjoy the moment and spread your happiness to others.",
        "sad": "It's okay to feel sad, try to engage in activities that lift your mood.",
        "surprise": "Take a moment to process the unexpected and stay mindful of your reaction."
    }.get(emotion, "Try to reflect on your emotions and breathe deeply.")
 
def get_reflection_prompt(emotion):
    return {
        "angry": "What triggered this anger, and how can you address it calmly?",
        "fear": "What is causing your fear, and how can you overcome it?",
        "happy": "What are you grateful for right now, and how can you spread positivity?",
        "sad": "What steps can you take to move forward and improve your mood?",
        "surprise": "How can you adapt to unexpected changes and stay positive?"
    }.get(emotion, "Reflect on your feelings and how you can improve your emotional well-being.")
 
# ---------- AUDIO VALIDATION ----------
def validate_audio_file(audio_path: str) -> bool:
    """Validate WAV file format and basic properties."""
    try:
        with contextlib.closing(wave.open(audio_path, 'r')) as audio_file:
            if audio_file.getnchannels() not in [1, 2]:
                return False
            if audio_file.getframerate() < 8000:
                return False
            if audio_file.getnframes() == 0:
                return False
        return True
    except (wave.Error, EOFError):
        return False
 
def save_audio_data(audio_data: np.ndarray, sample_rate: int, path: str) -> None:
    """Save numpy audio array to WAV file."""
    sf.write(path, audio_data, sample_rate)
 
def analyze_emotion(text: str, image, audio) -> Tuple:
    audio_path = None
    temp_file = None
    
    try:
        # Handle empty or None inputs
        if not audio and not text and image is None:
            raise ValueError("Please provide at least one input (text, image, or audio)")
            
        # Handle Gradio audio input
        if audio:
            # Create a temporary file with a .wav extension
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            audio_path = temp_file.name
            temp_file.close()
            
            if isinstance(audio, tuple):
                # Gradio audio component returns (sample_rate, audio_data)
                sample_rate, audio_data = audio
                if isinstance(audio_data, np.ndarray):
                    # Ensure audio data is in the correct format
                    if len(audio_data.shape) > 1:
                        # Convert stereo to mono if necessary
                        audio_data = np.mean(audio_data, axis=1)
                    save_audio_data(audio_data, sample_rate, audio_path)
                else:
                    raise ValueError("Invalid audio data format")
            # Handle file path input
            elif isinstance(audio, str) and os.path.exists(audio):
                if not audio.lower().endswith('.wav'):
                    raise ValueError("Only WAV files are supported")
                audio_path = audio
            else:
                raise ValueError("Unsupported audio input format")

            if not validate_audio_file(audio_path):
                raise ValueError("Invalid WAV file format")

        final_label, final_confidence, text_label, text_conf, voice_label, voice_conf, image_label, image_conf = fusion_predict(text, audio_path, image)
        print("DEBUG - Voice confidence type and value:", type(voice_conf), voice_conf)
        return final_label, {
            "Text": {"Predicted": text_label, "Confidence": float(text_conf)},
            "Voice": {"Predicted": voice_label, "Confidence": float(voice_conf)},
            "Image": {"Predicted": image_label, "Confidence": float(image_conf)}
        }, get_mindfulness_suggestions(final_label), get_reflection_prompt(final_label)

    except Exception as e:
        raise ValueError(f"Processing failed: {str(e)}. Please ensure your inputs are valid.")

    finally:
        # Clean up the temporary file
        if temp_file and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {audio_path}: {e}")

# Gradio Interface Configuration
interface = gr.Interface(
    fn=analyze_emotion,
    inputs=[
        gr.Textbox(label="Enter journal text"),
        gr.Image(label="Upload an image"),
        gr.Audio(
            label="Upload your voice",
            type="numpy",
            format="wav"
        )
    ],
    outputs=[
        gr.Textbox(label="Detected Emotion"),
        gr.JSON(label="Confidence Scores"),
        gr.Textbox(label="Mindfulness Suggestions"),
        gr.Textbox(label="Reflection Prompt")
    ],
    title="AI-powered Mood Journal & Emotion Tracker",
    description="Analyze your emotions through text, image, and voice inputs"
)
 
interface.launch()