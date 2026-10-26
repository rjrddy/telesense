"""Keyword-based psychiatric evaluation detector."""

import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


class PsychiatricKeywordDetector:
    """
    Detect psychiatric concerns based on keywords and patterns.
    """
    
    def __init__(self):
        """
        Initialize keyword patterns for psychiatric evaluation.
        Input:
            None
        """
        self._init_keyword_patterns()
    
    def _init_keyword_patterns(self):
        """
        Initialize keyword patterns for different psychiatric concerns.
        Input:
            None

        Output:
            None
        """
        
        # CRITICAL - Immediate risk indicators
        self.critical_patterns = {
            "suicide_ideation": {
                "keywords": [
                    "kill myself", "end my life", "suicide", "kill myself", "end it all",
                    "want to die", "better off dead", "not worth living", "no reason to live",
                    "take my life", "end everything", "don't want to be here", "wish I was dead"
                ],
                "severity": "high",
                "concern_name": "Suicide_Ideation",
                "weight": 1.0
            },
            "self_harm": {
                "keywords": [
                    "cut myself", "hurt myself", "self harm", "self-injury", "burn myself",
                    "scratch myself", "harm myself", "injure myself"
                ],
                "severity": "high",
                "concern_name": "Self_Harm",
                "weight": 1.0
            },
            "homicidal_ideation": {
                "keywords": [
                    "kill them", "hurt them", "want to hurt", "going to hurt", "make them pay",
                    "revenge", "get back at", "harm someone"
                ],
                "severity": "high",
                "concern_name": "Homicidal_Ideation",
                "weight": 1.0
            }
        }
        
        # HIGH PRIORITY - Serious mental health concerns
        self.high_priority_patterns = {
            "depression_severe": {
                "keywords": [
                    "hopeless", "worthless", "empty", "numb", "can't feel", "no emotion",
                    "nothing matters", "pointless", "meaningless", "no purpose",
                    "can't get out of bed", "can't function", "can't cope", "overwhelmed",
                    "constant sadness", "deep sadness", "profound sadness"
                ],
                "severity": "high",
                "concern_name": "Severe_Depression",
                "weight": 0.9
            },
            "anxiety_severe": {
                "keywords": [
                    "panic attack", "can't breathe", "heart racing", "chest tight", "hyperventilating",
                    "constant worry", "can't stop worrying", "overwhelming fear", "terrified",
                    "anxiety attack", "panic", "feeling trapped", "claustrophobic"
                ],
                "severity": "high",
                "concern_name": "Severe_Anxiety",
                "weight": 0.9
            },
            "psychosis": {
                "keywords": [
                    "hear voices", "seeing things", "voices telling me", "someone is watching",
                    "people following me", "conspiracy", "mind control", "thoughts not my own",
                    "reality feels off", "things aren't real", "delusional", "paranoid",
                    "hallucinations", "delusions"
                ],
                "severity": "high",
                "concern_name": "Psychosis",
                "weight": 0.95
            },
            "trauma_ptsd": {
                "keywords": [
                    "flashback", "nightmare", "trauma", "triggered", "ptsd", "post traumatic",
                    "can't forget", "reliving", "traumatic event", "abuse", "assault",
                    "violence", "accident", "disaster", "war", "combat"
                ],
                "severity": "high",
                "concern_name": "Trauma_PTSD",
                "weight": 0.85
            }
        }
        
        # MEDIUM PRIORITY - Significant concerns
        self.medium_priority_patterns = {
            "substance_use": {
                "keywords": [
                    "drinking too much", "can't stop drinking", "alcohol problem", "drug use",
                    "using drugs", "addiction", "substance abuse", "overdose", "withdrawal",
                    "need a drink", "need drugs", "can't function without"
                ],
                "severity": "medium",  # High Concern
                "concern_name": "Substance_Use",
                "weight": 0.7
            },
            "eating_disorder": {
                "keywords": [
                    "not eating", "can't eat", "won't eat", "throwing up", "purging",
                    "binge eating", "starving myself", "anorexia", "bulimia",
                    "body image", "hate my body", "too fat", "too thin"
                ],
                "severity": "medium",  # High Concern
                "concern_name": "Eating_Disorder",
                "weight": 0.7
            },
            "sleep_disorder": {
                "keywords": [
                    "can't sleep", "insomnia", "not sleeping", "sleeping too much",
                    "nightmares", "night terrors", "sleep paralysis", "exhausted",
                    "always tired", "never rested"
                ],
                "severity": "medium",  # High Concern
                "concern_name": "Sleep_Disorder",
                "weight": 0.6
            },
            "social_isolation": {
                "keywords": [
                    "no friends", "alone", "isolated", "nobody cares", "no one understands",
                    "can't connect", "social anxiety", "avoid people", "stay home",
                    "don't want to see anyone", "withdrawn"
                ],
                "severity": "medium",  # High Concern
                "concern_name": "Social_Isolation",
                "weight": 0.6
            },
            "grief_loss": {
                "keywords": [
                    "lost someone", "death of", "grieving", "mourning", "can't move on",
                    "miss them", "they're gone", "passed away", "died"
                ],
                "severity": "medium",  # High Concern
                "concern_name": "Grief_Loss",
                "weight": 0.65
            },
            "anger_aggression": {
                "keywords": [
                    "so angry", "rage", "furious", "lashing out", "can't control anger",
                    "violent thoughts", "want to break things", "explosive anger",
                    "temper", "outburst"
                ],
                "severity": "medium",  # High Concern
                "concern_name": "Anger_Aggression",
                "weight": 0.7
            }
        }
        
        # LOWER PRIORITY - Still important but less urgent
        self.lower_priority_patterns = {
            "stress": {
                "keywords": [
                    "stressed", "overwhelmed", "pressure", "too much", "can't handle",
                    "burnout", "exhausted", "stressed out"
                ],
                "severity": "low",
                "concern_name": "Stress",
                "weight": 0.5
            },
            "relationship_issues": {
                "keywords": [
                    "relationship problems", "fighting", "conflict", "breakup", "divorce",
                    "marital problems", "family issues", "relationship stress"
                ],
                "severity": "low",
                "concern_name": "Relationship_Issues",
                "weight": 0.5
            },
            "work_school_stress": {
                "keywords": [
                    "work stress", "job stress", "school stress", "academic pressure",
                    "can't keep up", "falling behind", "workplace issues"
                ],
                "severity": "low",
                "concern_name": "Work_School_Stress",
                "weight": 0.4
            }
        }
        
        # Combine all patterns
        self.all_patterns = {
            **self.critical_patterns,
            **self.high_priority_patterns,
            **self.medium_priority_patterns,
            **self.lower_priority_patterns
        }
    
    def detect_keywords(self, text: str, case_sensitive: bool = False) -> List[Dict]:
        """
        Detect psychiatric keywords in text.
        Input:
            text - Input text to analyze
            case_sensitive - Whether to match case (default: False)

        Output:
            List of detected concerns with:
                - concern_name: Name of concern
                - severity: "high", "medium", or "low"
                - confidence: Confidence score (0-1)
                - matched_keywords: List of matched keywords
                - weight: Weight for this concern
        """
        if not text or len(text.strip()) < 3:
            return []
        
        text_to_search = text if case_sensitive else text.lower()
        detected_concerns = defaultdict(lambda: {
            "concern_name": "",
            "severity": "low",
            "confidence": 0.0,
            "matched_keywords": [],
            "weight": 0.0
        })
        
        # Search for patterns
        for pattern_name, pattern_info in self.all_patterns.items():
            keywords = pattern_info["keywords"]
            matched_keywords = []
            
            for keyword in keywords:
                keyword_lower = keyword.lower() if not case_sensitive else keyword
                # Use word boundaries for better matching
                pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                matches = re.findall(pattern, text_to_search, re.IGNORECASE)
                
                if matches:
                    matched_keywords.extend(matches)
            
            if matched_keywords:
                # Calculate confidence based on number of matches and pattern weight
                num_matches = len(matched_keywords)
                base_confidence = min(0.5 + (num_matches * 0.15), 1.0)
                confidence = base_confidence * pattern_info["weight"]
                
                # Update or create concern entry
                if pattern_info["concern_name"] not in detected_concerns or \
                   confidence > detected_concerns[pattern_info["concern_name"]]["confidence"]:
                    detected_concerns[pattern_info["concern_name"]] = {
                        "concern_name": pattern_info["concern_name"],
                        "severity": pattern_info["severity"],
                        "confidence": confidence,
                        "matched_keywords": list(set(matched_keywords)),
                        "weight": pattern_info["weight"]
                    }
        
        # Convert to list and sort by confidence
        results = list(detected_concerns.values())
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return results
    
    def detect_keywords_for_segments(self, segments: List[Dict], 
                                     min_confidence: float = 0.4) -> List[Dict]:
        """
        Detect keywords for multiple text segments with timestamps.
        Input:
            segments - List of dicts with 'text', 'start_time', 'end_time'
            min_confidence - Minimum confidence threshold

        Output:
            List of keyword detection results with timestamps
        """
        results = []
        
        for segment in segments:
            text = segment.get("text", "")
            if not text:
                continue
            
            keyword_results = self.detect_keywords(text)
            
            # Filter by confidence
            filtered_results = [
                r for r in keyword_results 
                if r["confidence"] >= min_confidence
            ]
            
            if filtered_results:
                # Use the highest confidence result
                top_result = filtered_results[0]
                results.append({
                    "start_time": segment.get("start_time", 0.0),
                    "end_time": segment.get("end_time", 0.0),
                    "concern_name": top_result["concern_name"],
                    "severity": top_result["severity"],
                    "confidence": top_result["confidence"],
                    "matched_keywords": top_result["matched_keywords"],
                    "modality": "text",
                    "detection_method": "keyword"
                })
        
        return results
    
    def get_concern_priority(self, concern_name: str) -> str:
        """
        Get priority level for a concern name.
        Input:
            concern_name - Name of concern

        Output:
            "critical", "high", "medium", or "low"
        """
        for pattern_name, pattern_info in self.critical_patterns.items():
            if pattern_info["concern_name"] == concern_name:
                return "critical"
        
        for pattern_name, pattern_info in self.high_priority_patterns.items():
            if pattern_info["concern_name"] == concern_name:
                return "high"
        
        for pattern_name, pattern_info in self.medium_priority_patterns.items():
            if pattern_info["concern_name"] == concern_name:
                return "medium"  # High Concern
        
        return "low"
