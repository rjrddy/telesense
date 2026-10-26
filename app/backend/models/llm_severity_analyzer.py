"""LLM-based severity analysis for multimodal behavioral concerns."""

import os
from typing import Dict, List, Optional, Tuple
import json

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class LLMSeverityAnalyzer:
    """
    Use LLM to analyze multimodal features and assign severity levels.
    """
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "openai"):
        """
        Initialize LLM severity analyzer.
        Input:
            api_key - API key for LLM service (if None, uses environment variable)
            provider - "openai" or "anthropic"
        """
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        
        if provider == "openai" and OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        elif provider == "anthropic" and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
        else:
            self.client = None
            print("Warning: LLM client not available. Install openai or anthropic package and set API key.")
    
    def analyze_severity(
        self,
        text: str,
        audio_features: Optional[Dict] = None,
        visual_features: Optional[Dict] = None,
        concern_type: str = "Unknown"
    ) -> Dict:
        """
        Analyze severity using LLM based on multimodal features.
        Input:
            text - Transcribed text segment
            audio_features - Audio features (tone, prosody, etc.)
            visual_features - Visual features (facial expressions, body language)
            concern_type - Type of concern detected

        Output:
            Dictionary with:
                - severity: "high", "medium", or "low"
                - reasoning: Explanation for severity level
                - confidence: Confidence score (0-1)
        """
        if not self.client:
            # Fallback to rule-based if LLM not available
            return self._rule_based_severity(text, audio_features, visual_features, concern_type)
        
        # Build prompt with multimodal context
        prompt = self._build_prompt(text, audio_features, visual_features, concern_type)
        
        try:
            if self.provider == "openai":
                return self._analyze_with_openai(prompt, concern_type)
            elif self.provider == "anthropic":
                return self._analyze_with_anthropic(prompt, concern_type)
        except Exception as e:
            print(f"Error in LLM severity analysis: {e}")
            # Fallback to rule-based
            return self._rule_based_severity(text, audio_features, visual_features, concern_type)
    
    def _build_prompt(
        self,
        text: str,
        audio_features: Optional[Dict],
        visual_features: Optional[Dict],
        concern_type: str
    ) -> str:
        """
        Build prompt for LLM analysis.
        Input:
            text - segment transcript
            audio_features - segment audio features
            visual_features - segment visual features
            concern_type - flagged concern type

        Output:
            Prompt string
        """
        
        prompt = f"""You are a psychiatric evaluation assistant. Analyze the following multimodal data to determine the severity level (high, medium, or low) of a behavioral concern.

Concern Type: {concern_type}

Text Content: "{text}"

"""
        
        if audio_features:
            prompt += f"""Audio Features:
- Tone: {audio_features.get('tone', 'N/A')}
- Speech Rate: {audio_features.get('speech_rate', 'N/A')}
- Pitch Variance: {audio_features.get('pitch_variance', 'N/A')}
- Volume: {audio_features.get('volume', 'N/A')}
- Prosody: {audio_features.get('prosody', 'N/A')}

"""
        
        if visual_features:
            prompt += f"""Visual Features:
- Facial Expression: {visual_features.get('facial_expression', 'N/A')}
- Body Language: {visual_features.get('body_language', 'N/A')}
- Eye Contact: {visual_features.get('eye_contact', 'N/A')}
- Gestures: {visual_features.get('gestures', 'N/A')}

"""
        
        prompt += """Based on the text content, audio characteristics, and visual cues, determine the severity level:

HIGH: Immediate concern requiring urgent attention (e.g., suicidal ideation, severe distress, acute crisis)
MEDIUM: Moderate concern requiring follow-up (e.g., significant distress, concerning patterns)
LOW: Mild concern or normal variation (e.g., minor stress indicators, typical emotional expression)

Respond in JSON format:
{
  "severity": "high|medium|low",
  "reasoning": "Brief explanation",
  "confidence": 0.0-1.0
}"""
        
        return prompt
    
    def _analyze_with_openai(self, prompt: str, concern_type: str) -> Dict:
        """
        Analyze using OpenAI API.
        Input:
            prompt - prompt string
            concern_type - flagged concern type

        Output:
            Dictionary with severity, reasoning and confidence
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Use cost-effective model
            messages=[
                {"role": "system", "content": "You are a psychiatric evaluation assistant. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return {
            "severity": result.get("severity", "low").lower(),
            "reasoning": result.get("reasoning", ""),
            "confidence": float(result.get("confidence", 0.5))
        }
    
    def _analyze_with_anthropic(self, prompt: str, concern_type: str) -> Dict:
        """
        Analyze using Anthropic Claude API.
        Input:
            prompt - prompt string
            concern_type - flagged concern type

        Output:
            Dictionary with severity, reasoning and confidence
        """
        message = self.client.messages.create(
            model="claude-3-haiku-20240307",  # Cost-effective model
            max_tokens=500,
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        content = message.content[0].text
        # Try to extract JSON from response
        try:
            # Find JSON in response
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                result = json.loads(content[start:end])
            else:
                raise ValueError("No JSON found")
        except:
            # Fallback parsing
            result = {"severity": "low", "reasoning": content, "confidence": 0.5}
        
        return {
            "severity": result.get("severity", "low").lower(),
            "reasoning": result.get("reasoning", ""),
            "confidence": float(result.get("confidence", 0.5))
        }
    
    def _rule_based_severity(
        self,
        text: str,
        audio_features: Optional[Dict],
        visual_features: Optional[Dict],
        concern_type: str
    ) -> Dict:
        """
        Fallback rule-based severity analysis.
        Input:
            text - segment transcript
            audio_features - segment audio features
            visual_features - segment visual features
            concern_type - flagged concern type

        Output:
            Dictionary with severity, reasoning and confidence
        """
        severity_score = 0.0
        reasoning_parts = []
        
        # Text-based indicators
        text_lower = text.lower()
        high_indicators = [
            "suicide", "kill myself", "end it all", "want to die", "no point",
            "hopeless", "worthless", "can't go on", "give up"
        ]
        medium_indicators = [
            "depressed", "anxious", "overwhelmed", "stressed", "worried",
            "difficult", "struggling", "hard time", "feeling down"
        ]
        
        for indicator in high_indicators:
            if indicator in text_lower:
                severity_score += 0.4
                reasoning_parts.append(f"High-risk language detected: '{indicator}'")
        
        for indicator in medium_indicators:
            if indicator in text_lower:
                severity_score += 0.2
                reasoning_parts.append(f"Concerning language: '{indicator}'")
        
        # Audio-based indicators
        if audio_features:
            if audio_features.get("tone") in ["monotone", "flat", "strained"]:
                severity_score += 0.2
                reasoning_parts.append("Concerning audio tone detected")
            if audio_features.get("speech_rate", 1.0) < 0.7:
                severity_score += 0.1
                reasoning_parts.append("Slow speech rate")
        
        # Visual-based indicators
        if visual_features:
            if visual_features.get("facial_expression") in ["sad", "distressed", "anxious"]:
                severity_score += 0.2
                reasoning_parts.append("Concerning facial expression")
            if visual_features.get("eye_contact") == "avoiding":
                severity_score += 0.1
                reasoning_parts.append("Avoiding eye contact")
        
        # Determine severity
        if severity_score >= 0.6:
            severity = "high"
        elif severity_score >= 0.3:
            severity = "medium"  # High Concern
        else:
            severity = "low"
        
        return {
            "severity": severity,
            "reasoning": "; ".join(reasoning_parts) if reasoning_parts else "No significant indicators detected",
            "confidence": min(1.0, severity_score)
        }
    
    def batch_analyze_severities(
        self,
        segments: List[Dict],
        audio_features_list: Optional[List[Dict]] = None,
        visual_features_list: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Analyze severity for multiple segments.
        Input:
            segments - List of segments with text and concern_type
            audio_features_list - Optional list of audio features per segment
            visual_features_list - Optional list of visual features per segment

        Output:
            List of severity analysis results
        """
        results = []
        for i, segment in enumerate(segments):
            audio_feat = audio_features_list[i] if audio_features_list and i < len(audio_features_list) else None
            visual_feat = visual_features_list[i] if visual_features_list and i < len(visual_features_list) else None
            
            result = self.analyze_severity(
                text=segment.get("text", ""),
                audio_features=audio_feat,
                visual_features=visual_feat,
                concern_type=segment.get("concern_type", "Unknown")
            )
            results.append(result)
        
        return results
