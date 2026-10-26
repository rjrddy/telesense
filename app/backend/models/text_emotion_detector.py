"""Text emotion detection using multi-label RoBERTa model."""

import torch
import numpy as np
from typing import List, Dict, Optional

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Install with: pip install transformers")


class TextEmotionDetector:
    """
    Detect emotions from text using multi-label classification.
    """
    
    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions", device: Optional[str] = None):
        """
        Initialize text emotion detector.
        Input:
            model_name - Hugging Face model name for emotion recognition
            device - Device to run on ('cpu', 'cuda', 'mps')
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers")
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # MPS might have issues with some models, use CPU for text
        if self.device == "mps":
            self.text_device = "cpu"
        else:
            self.text_device = self.device
        
        # Load model
        print(f"Loading text emotion model: {model_name} on {self.text_device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.text_device)
            self.model.eval()
            
            # Get emotion labels from model config
            if hasattr(self.model.config, 'id2label'):
                self.emotion_labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
            else:
                # Default labels for go_emotions model
                self.emotion_labels = [
                    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
                    "confusion", "curiosity", "desire", "disappointment", "disapproval",
                    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
                    "joy", "love", "nervousness", "optimism", "pride", "realization",
                    "relief", "remorse", "sadness", "surprise", "neutral"
                ]
            
            print(f"Successfully loaded text emotion model with {len(self.emotion_labels)} emotion classes")
        except Exception as e:
            print(f"Error loading text emotion model: {e}")
            raise
        
        # Map emotions to behavioral concerns
        self._init_emotion_to_concern_mapping()
    
    def _init_emotion_to_concern_mapping(self):
        """
        Initialize mapping from emotions to behavioral concerns.
        Input:
            None

        Output:
            None
        """
        # High concern emotions
        high_concern_emotions = {
            "sadness": "Depression",
            "grief": "Depression",
            "despair": "Depression",
            "anger": "Agitation",
            "annoyance": "Agitation",
            "disgust": "Negative_Emotion",
            "fear": "Anxiety",
            "nervousness": "Anxiety",
            "remorse": "Negative_Emotion",
            "disappointment": "Depression",
            "disapproval": "Negative_Emotion"
        }
        
        # Medium concern emotions
        medium_concern_emotions = {
            "confusion": "Cognitive_Concern",
            "embarrassment": "Social_Concern",
            "caring": "Positive_Emotion",  # Can indicate caregiver stress
            "curiosity": "Neutral"
        }
        
        # Low concern / positive emotions
        positive_emotions = {
            "joy": "Positive_Emotion",
            "amusement": "Positive_Emotion",
            "excitement": "Positive_Emotion",
            "gratitude": "Positive_Emotion",
            "love": "Positive_Emotion",
            "optimism": "Positive_Emotion",
            "pride": "Positive_Emotion",
            "relief": "Positive_Emotion",
            "admiration": "Positive_Emotion",
            "approval": "Positive_Emotion",
            "surprise": "Neutral",
            "neutral": "Neutral",
            "realization": "Neutral"
        }
        
        self.emotion_to_concern = {**high_concern_emotions, **medium_concern_emotions, **positive_emotions}
    
    def detect_emotions(self, text: str, return_all: bool = False, 
                       threshold: float = 0.3) -> Dict:
        """
        Detect emotions from text (multi-label).
        Input:
            text - Input text
            return_all - If True, return all emotions above threshold; if False, return top emotions
            threshold - Probability threshold for including emotions

        Output:
            Dictionary with:
                - emotions: Dict of emotion -> probability (above threshold)
                - top_emotions: List of top emotions with scores
                - concerns: List of behavioral concerns mapped from emotions
                - dominant_emotion: Emotion with highest probability
                - confidence: Confidence score
        """
        if not text or len(text.strip()) < 1:
            return {
                "emotions": {},
                "top_emotions": [],
                "concerns": [],
                "dominant_emotion": "neutral",
                "confidence": 0.0
            }
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.text_device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.sigmoid(logits)  # Multi-label uses sigmoid
            
            # Get probabilities
            probs_np = probs[0].cpu().numpy()
            
            # Get all emotions above threshold
            emotions = {}
            emotion_scores = []
            
            for i, label in enumerate(self.emotion_labels):
                if i < len(probs_np):
                    prob = float(probs_np[i])
                    if prob >= threshold:
                        emotions[label] = prob
                        emotion_scores.append((label, prob))
            
            # Sort by probability
            emotion_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Get top emotions (up to 3)
            top_emotions = emotion_scores[:3] if not return_all else emotion_scores
            
            # Map to concerns
            concerns = self._map_emotions_to_concerns(emotion_scores)
            
            # Get dominant emotion
            if emotion_scores:
                dominant_emotion = emotion_scores[0][0]
                confidence = emotion_scores[0][1]
            else:
                dominant_emotion = "neutral"
                confidence = 0.0
            
            return {
                "emotions": emotions,
                "top_emotions": [(emotion, score) for emotion, score in top_emotions],
                "concerns": concerns,
                "dominant_emotion": dominant_emotion,
                "confidence": confidence
            }
        except Exception as e:
            print(f"Error in text emotion detection: {e}")
            return {
                "emotions": {},
                "top_emotions": [],
                "concerns": [],
                "dominant_emotion": "neutral",
                "confidence": 0.0
            }
    
    def _map_emotions_to_concerns(self, emotion_scores: List[tuple]) -> List[str]:
        """
        Map detected emotions to behavioral concerns.
        Input:
            emotion_scores - List of (emotion, score) tuples

        Output:
            List of unique concern names
        """
        concerns = set()
        
        for emotion, score in emotion_scores:
            concern = self.emotion_to_concern.get(emotion, "Emotional_Concern")
            if concern != "Neutral" and concern != "Positive_Emotion":
                concerns.add(concern)
        
        return list(concerns)
    
    def detect_emotions_for_segments(self, segments: List[Dict], threshold: float = 0.3) -> List[Dict]:
        """
        Detect emotions for multiple text segments with timestamps.
        Input:
            segments - List of dicts with 'text', 'start_time', 'end_time'
            threshold - Probability threshold for including emotions

        Output:
            List of emotion detection results with timestamps
        """
        results = []
        
        for segment in segments:
            text = segment.get("text", "")
            if not text:
                continue
            
            emotion_result = self.detect_emotions(text, threshold=threshold)
            
            results.append({
                "start_time": segment.get("start_time", 0.0),
                "end_time": segment.get("end_time", 0.0),
                **emotion_result
            })
        
        return results
