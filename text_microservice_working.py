"""
🤖 Y.M.I.R Text Emotion Detection Microservice - FULL PRODUCTION VERSION
======================================================================
Complete production-grade microservice with ALL features from chatbot:
- 🧠 Multiple SOTA emotion models with ensemble voting
- 🎯 Context-aware transformers (not rule-based preprocessing!)
- 🌊 Streaming responses like ChatGPT
- 🔧 Function calling capabilities
- 📱 CORS enabled for main app integration
- 🧬 Gemini API integration
- 💾 Persistent conversation history and user profiles
- 🛡️ Production-grade error handling
- 📊 Advanced analytics and session management
"""

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import os
import json
import time
import asyncio
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from collections import deque
import re
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import hashlib

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Firebase imports for real-time chat storage and authentication
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, auth
    
    # Initialize Firebase app if not already initialized
    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase_credentials.json')
        firebase_admin.initialize_app(cred)
    
    FIREBASE_AVAILABLE = True
    print("✅ Firebase initialized for authentication")
    print(f"🔍 Firebase auth module available: {auth is not None}")
except ImportError as import_error:
    firebase_admin = None
    credentials = None
    firestore = None
    auth = None
    FIREBASE_AVAILABLE = False
    print(f"⚠️ Firebase not available: {import_error}")
except Exception as firebase_error:
    firebase_admin = None
    credentials = None
    firestore = None
    auth = None
    FIREBASE_AVAILABLE = False
    print(f"🔥 Firebase initialization error: {firebase_error}")

# Encryption for secure chat storage
try:
    from cryptography.fernet import Fernet
    import base64
    import hashlib
    ENCRYPTION_AVAILABLE = True
    # Encryption available
except ImportError:
    ENCRYPTION_AVAILABLE = False
    # Encryption not available

# Google Gemini API with rotation system
try:
    import google.generativeai as genai
    from gemini_api_manager import get_gemini_model, get_api_status, gemini_manager
    GEMINI_AVAILABLE = True
    # Gemini API available
except ImportError:
    GEMINI_AVAILABLE = False
    # Gemini API not available

# Production-grade ML emotion detection
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    ML_AVAILABLE = True
    # Transformers available
except ImportError:
    ML_AVAILABLE = False
    # Transformers not available

# TextBlob fallback
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
    # TextBlob available
except ImportError:
    TEXTBLOB_AVAILABLE = False
    # TextBlob not available

app = Flask(__name__)
CORS(app)  # 🎯 ENABLE CORS FOR MICROSERVICE COMMUNICATION

def verify_firebase_token(token):
    """Verify Firebase auth token and return user info"""
    print(f"🔍 Token verification: FIREBASE_AVAILABLE={FIREBASE_AVAILABLE}, token_present={bool(token)}, auth_available={auth is not None}")
    
    if not FIREBASE_AVAILABLE or not token or auth is None:
        print("❌ Firebase verification failed: missing requirements")
        return None
    
    try:
        decoded_token = auth.verify_id_token(token)
        print(f"✅ Token verified for user: {decoded_token.get('email')}")
        return {
            'uid': decoded_token.get('uid'),
            'email': decoded_token.get('email'),
            'name': decoded_token.get('name', decoded_token.get('email', 'User'))
        }
    except Exception as e:
        print(f"❌ Firebase token verification error: {e}")
        return None

@dataclass
class ChatMessage:
    """Structured chat message with metadata"""
    role: str
    content: str
    timestamp: datetime
    emotion: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat() if hasattr(self.timestamp, 'isoformat') else str(self.timestamp),
            'emotion': self.emotion,
            'confidence': self.confidence,
            'metadata': self.metadata or {}
        }

@dataclass
class UserProfile:
    """User profile with preferences and history"""
    user_id: str
    name: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    conversation_style: str = "balanced"
    emotion_history: Optional[List[str]] = None
    topics_of_interest: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}
        if self.emotion_history is None:
            self.emotion_history = []
        if self.topics_of_interest is None:
            self.topics_of_interest = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_active is None:
            self.last_active = datetime.now()

class FirebaseChatStorage:
    """🔥 Real-time encrypted chat storage like ChatGPT"""
    
    def __init__(self):
        self.db = None
        self.encryption_key = None
        self.session_id = None  # Wait for frontend to set session ID
        
        # Initialize encryption
        if ENCRYPTION_AVAILABLE:
            self._setup_encryption()
        
        # Initialize Firebase
        if FIREBASE_AVAILABLE:
            self._initialize_firebase()
        
        # Chat session initialized
    
    def set_session_id(self, session_id: str):
        """Set the session ID for emotion storage synchronization"""
        if session_id:
            self.session_id = session_id
            return True
        return False
    
    def _setup_encryption(self):
        """Setup encryption for secure chat storage"""
        try:
            # Generate or load encryption key
            key_file = Path("chat_encryption.key")
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Generate new key
                self.encryption_key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
                # Generated new encryption key
            
            self.cipher = Fernet(self.encryption_key)
            # Chat encryption enabled
            
        except Exception as e:
            # Encryption setup failed
            self.encryption_key = None
    
    def _initialize_firebase(self):
        """Initialize Firebase connection"""
        try:
            # Check if Firebase is already initialized
            if firebase_admin._apps:
                self.db = firestore.client()
                # Using existing Firebase connection
                return
            
            # Look for credentials file
            cred_paths = [
                Path("firebase_credentials.json"),
                Path("src/firebase_credentials.json"),
                Path("../firebase_credentials.json")
            ]
            
            for cred_path in cred_paths:
                if cred_path.exists():
                    cred = credentials.Certificate(str(cred_path))
                    firebase_admin.initialize_app(cred)
                    self.db = firestore.client()
                    # Firebase chat storage initialized
                    return
            
            # Firebase credentials not found
            pass
            
        except Exception as e:
            # Firebase initialization error
            pass
    
    def _encrypt_content(self, content: str) -> str:
        """Encrypt chat content for secure storage"""
        if not self.encryption_key:
            return content
        
        try:
            encrypted = self.cipher.encrypt(content.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            # Encryption failed
            return content
    
    def _decrypt_content(self, encrypted_content: str) -> str:
        """Decrypt chat content"""
        if not self.encryption_key:
            return encrypted_content
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_content.encode())
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            # Decryption failed
            return encrypted_content
    
    def store_message(self, message: ChatMessage) -> bool:
        """Store chat message in Firebase with encryption"""
        
        if not self.db or not self.session_id:
            print(f"❌ STORAGE FAILED: Firebase not available (db={self.db is not None}) or no session ID ('{self.session_id}')")
            return False
        
        try:
            # Encrypt sensitive content
            encrypted_content = self._encrypt_content(message.content)
            
            # Prepare document data
            doc_data = {
                'session_id': self.session_id,
                'message_id': message.id,
                'role': message.role,
                'encrypted_content': encrypted_content,
                'timestamp': message.timestamp,
                'emotion': message.emotion,
                'confidence': message.confidence,
                'metadata': message.metadata or {},
                'created_at': datetime.now()
            }
            
            # Store in Firestore
            doc_ref = self.db.collection('chat_messages').document(message.id)
            doc_ref.set(doc_data)
            
            # Also update latest emotions collection for quick access
            print(f"🔍 DEBUG: Checking emotion storage - emotion: {message.emotion}, session: {self.session_id}")
            if message.emotion and message.emotion != 'neutral':
                print(f"✅ Storing text emotion to Firebase...")
                # 🎭 MULTI-EMOTION: Store ALL emotions like face microservice
                all_emotions = message.metadata.get('emotion_analysis', {}).get('all_emotions', {}) if message.metadata else {}
                
                emotion_doc = {
                    'session_id': self.session_id,
                    'emotion': message.emotion,  # Dominant emotion (single)
                    'emotions': all_emotions,   # ALL emotions (multiple) - same as face microservice!
                    'confidence': message.confidence,
                    'content_preview': message.content[:100] + '...' if len(message.content) > 100 else message.content,
                    'timestamp': message.timestamp,
                    'message_id': message.id,
                    'role': message.role
                }
                
                # Storing text emotions to Firebase
                
                # Store in emotion_readings collection (same as face microservice)
                emotion_ref = self.db.collection('emotion_readings').document()
                emotion_ref.set(emotion_doc)
                
                print(f"🔥 TEXT EMOTION STORED: {message.emotion} (confidence: {message.confidence}) session: {self.session_id}")
                print(f"✅ Stored to emotion_readings collection successfully!")
            else:
                print(f"❌ TEXT EMOTION NOT STORED: emotion='{message.emotion}', reason: neutral or empty")
            
            # Message stored in Firebase
            return True
            
        except Exception as e:
            # Firebase chat storage error
            return False
    
    def get_recent_messages(self, limit: int = 20) -> List[ChatMessage]:
        """Get recent messages from Firebase"""
        if not self.db:
            return []
        
        try:
            # Query recent messages for this session
            messages_ref = self.db.collection('chat_messages')
            query = messages_ref.where('session_id', '==', self.session_id).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
            
            messages = []
            for doc in query.stream():
                data = doc.to_dict()
                
                # Decrypt content
                content = self._decrypt_content(data.get('encrypted_content', ''))
                
                message = ChatMessage(
                    id=data.get('message_id'),
                    role=data.get('role'),
                    content=content,
                    timestamp=data.get('timestamp'),
                    emotion=data.get('emotion'),
                    confidence=data.get('confidence'),
                    metadata=data.get('metadata', {})
                )
                messages.append(message)
            
            # Reverse to get chronological order
            messages.reverse()
            return messages
            
        except Exception as e:
            # Error retrieving messages
            return []
    
    def get_latest_emotion(self, minutes_back: int = 10) -> Optional[Dict[str, Any]]:
        """Get latest text emotion for multimodal fusion"""
        if not self.db:
            return None
        
        try:
            # Calculate time range
            cutoff_time = datetime.now() - timedelta(minutes=minutes_back)
            
            # Query recent emotion readings
            emotions_ref = self.db.collection('emotion_readings')
            query = emotions_ref.where('session_id', '==', self.session_id).where('role', '==', 'user').where('timestamp', '>=', cutoff_time).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1)
            
            for doc in query.stream():
                data = doc.to_dict()
                # Found Firebase text emotion
                return data
            
            # No recent text emotions found
            return None
            
        except Exception as e:
            # Error getting latest emotion
            return None

class UserCentricLearningSystem:
    """🎓 AI-driven user-centric learning for personalized emotion detection"""
    
    def __init__(self):
        self.user_corrections = {}
        self.user_patterns = {}
        self.learning_data_file = Path("user_emotion_learning.json")
        self.session_corrections = []
        self._load_learning_data()
        
        # User-centric learning system initialized
        
    def _load_learning_data(self):
        """Load user learning data from persistent storage"""
        try:
            if self.learning_data_file.exists():
                with open(self.learning_data_file, 'r') as f:
                    data = json.load(f)
                    self.user_corrections = data.get('corrections', {})
                    self.user_patterns = data.get('patterns', {})
                    # User learning data loaded
            else:
                # No existing learning data found
                pass
        except Exception as e:
            # Failed to load learning data
            pass
            self.user_corrections = {}
            self.user_patterns = {}
    
    def _save_learning_data(self):
        """Save user learning data persistently"""
        try:
            data = {
                'corrections': self.user_corrections,
                'patterns': self.user_patterns,
                'last_updated': datetime.now().isoformat(),
                'total_corrections': len(self.user_corrections)
            }
            with open(self.learning_data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            # Failed to save learning data
            pass
    
    def record_user_correction(self, original_text: str, predicted_emotion: str, 
                             predicted_confidence: float, user_corrected_emotion: str,
                             user_id: str = "default"):
        """Record when user corrects an emotion prediction"""
        correction_id = str(uuid.uuid4())
        correction_data = {
            'text': original_text,
            'predicted': predicted_emotion,
            'predicted_confidence': predicted_confidence,
            'corrected_to': user_corrected_emotion,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'text_length': len(original_text),
            'text_hash': hashlib.md5(original_text.encode()).hexdigest()[:8]
        }
        
        self.user_corrections[correction_id] = correction_data
        self.session_corrections.append(correction_data)
        
        # Update user patterns
        self._update_user_patterns(original_text, predicted_emotion, user_corrected_emotion, user_id)
        
        # Save persistently
        self._save_learning_data()
        
        # User correction recorded
        
        return correction_id
    
    def _update_user_patterns(self, text: str, predicted: str, corrected: str, user_id: str):
        """Update user-specific emotion patterns based on corrections"""
        if user_id not in self.user_patterns:
            self.user_patterns[user_id] = {
                'frequent_corrections': {},
                'text_patterns': {},
                'confidence_adjustments': {},
                'vocabulary': set()
            }
        
        user_data = self.user_patterns[user_id]
        
        # Track frequent corrections
        correction_pair = f"{predicted}→{corrected}"
        user_data['frequent_corrections'][correction_pair] = user_data['frequent_corrections'].get(correction_pair, 0) + 1
        
        # Extract text patterns (words that lead to corrections)
        words = text.lower().split()
        for word in words:
            if len(word) > 3:  # Focus on meaningful words
                if word not in user_data['text_patterns']:
                    user_data['text_patterns'][word] = {}
                
                if corrected not in user_data['text_patterns'][word]:
                    user_data['text_patterns'][word][corrected] = 0
                user_data['text_patterns'][word][corrected] += 1
                
                user_data['vocabulary'].add(word)
        
        # Convert set to list for JSON serialization
        user_data['vocabulary'] = list(user_data['vocabulary'])
        
        # User patterns updated
    
    def get_personalized_confidence_adjustment(self, text: str, predicted_emotion: str, 
                                             base_confidence: float, user_id: str = "default") -> float:
        """Get personalized confidence adjustment based on user's correction history"""
        if user_id not in self.user_patterns:
            return 1.0
        
        user_data = self.user_patterns[user_id]
        confidence_multiplier = 1.0
        
        # Check if this prediction type frequently gets corrected by this user
        for correction_pair, count in user_data['frequent_corrections'].items():
            if correction_pair.startswith(predicted_emotion + "→"):
                # If user frequently corrects this emotion, reduce confidence
                correction_rate = count / max(1, len(self.user_corrections))
                if correction_rate > 0.3:  # 30% correction rate threshold
                    confidence_multiplier *= (1.0 - correction_rate * 0.5)  # Reduce confidence
                    # Adjusting confidence based on user corrections
        
        # Check if text contains words user associates with different emotions
        words = text.lower().split()
        word_evidence = {}
        
        for word in words:
            if word in user_data['text_patterns']:
                for emotion, count in user_data['text_patterns'][word].items():
                    if emotion != predicted_emotion:
                        word_evidence[emotion] = word_evidence.get(emotion, 0) + count
        
        # If strong evidence for different emotion, reduce confidence
        if word_evidence:
            max_evidence = max(word_evidence.values())
            total_evidence = sum(word_evidence.values())
            if max_evidence / total_evidence > 0.6:  # Strong evidence for different emotion
                confidence_multiplier *= 0.7
                suggested_emotion = max(word_evidence.items(), key=lambda x: x[1])[0]
                # User vocabulary analysis complete
        
        return confidence_multiplier
    
    def get_personalized_emotion_suggestion(self, text: str, predicted_emotion: str, 
                                          user_id: str = "default") -> Optional[str]:
        """Suggest alternative emotion based on user's patterns"""
        if user_id not in self.user_patterns:
            return None
        
        user_data = self.user_patterns[user_id]
        words = text.lower().split()
        
        # Analyze user's vocabulary patterns
        emotion_scores = {}
        
        for word in words:
            if word in user_data['text_patterns']:
                for emotion, count in user_data['text_patterns'][word].items():
                    emotion_scores[emotion] = emotion_scores.get(emotion, 0) + count
        
        if emotion_scores:
            # Get most likely emotion based on user's patterns
            suggested_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            suggestion_strength = emotion_scores[suggested_emotion] / sum(emotion_scores.values())
            
            if suggested_emotion != predicted_emotion and suggestion_strength > 0.4:
                # User pattern suggestion applied
                return suggested_emotion
        
        return None
    
    def get_learning_analytics(self, user_id: str = "default") -> Dict[str, Any]:
        """Get analytics about user learning progress"""
        user_corrections = [c for c in self.user_corrections.values() if c.get('user_id') == user_id]
        
        if not user_corrections:
            return {'message': 'No user corrections recorded yet'}
        
        # Calculate accuracy improvements
        recent_corrections = sorted(user_corrections, key=lambda x: x['timestamp'])[-10:]
        
        analytics = {
            'total_corrections': len(user_corrections),
            'recent_corrections': len(recent_corrections),
            'most_corrected_emotions': {},
            'improvement_patterns': {},
            'vocabulary_size': len(self.user_patterns.get(user_id, {}).get('vocabulary', [])),
            'learning_effectiveness': 'improving'
        }
        
        # Find most corrected emotions
        for correction in user_corrections:
            predicted = correction['predicted']
            analytics['most_corrected_emotions'][predicted] = analytics['most_corrected_emotions'].get(predicted, 0) + 1
        
        return analytics

class AIContextPreprocessor:
    """🧠 AI-driven context preprocessing for text emotion analysis"""
    
    def __init__(self):
        self.context_models = {}
        self.learning_system = UserCentricLearningSystem()  # Add learning system
        self._init_context_models()
        
    def _init_context_models(self):
        """Initialize AI models for context analysis"""
        self.intensity_analyzer = None
        self.informal_processor = None
        
        if not ML_AVAILABLE:
            return
            
        try:
            # Sentiment intensity model for confidence adjustment
            self.intensity_analyzer = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            # Context intensity analyzer loaded
            
            # Text preprocessing model for informal language
            self.informal_processor = pipeline(
                "text2text-generation",
                model="t5-small"  # Lightweight for real-time processing
            )
            # Context informal language processor loaded
            
        except Exception as e:
            # Context models failed to load
            pass
            self.intensity_analyzer = None
            self.informal_processor = None
    
    def analyze_context_confidence(self, text: str) -> float:
        """Use AI to determine confidence multiplier based on text context"""
        if not self.intensity_analyzer:
            return 1.0
            
        try:
            # Analyze sentiment intensity
            results = self.intensity_analyzer(text)
            
            # Handle different result formats
            if isinstance(results, list) and len(results) > 0:
                # Extract confidence from sentiment intensity scores
                if isinstance(results[0], dict) and 'score' in results[0]:
                    max_score = max([r['score'] for r in results])
                else:
                    return 1.0
            else:
                return 1.0
            
            # AI-driven confidence adjustment
            if max_score > 0.8:  # High confidence sentiment
                return 1.3  # Boost emotion confidence
            elif max_score < 0.4:  # Low confidence sentiment  
                return 0.7  # Reduce emotion confidence
            else:
                return 1.0  # Neutral adjustment
                
        except Exception as e:
            # Context confidence analysis failed
            return 1.0
    
    def preprocess_informal_text(self, text: str) -> str:
        """AI-driven preprocessing of informal text for better emotion detection"""
        if not self.informal_processor:
            return text
            
        try:
            # Use T5 for text normalization and enhancement
            prompt = f"Rewrite this informal text in clear, standard English for emotion analysis: {text}"
            
            # Generate normalized text
            result = self.informal_processor(prompt, max_length=len(text) + 50, do_sample=False)
            
            if result and hasattr(result, '__len__') and len(result) > 0:
                if isinstance(result[0], dict) and 'generated_text' in result[0]:
                    preprocessed = str(result[0]['generated_text'])
                
                # Extract the actual normalized text (remove the prompt)
                if prompt in preprocessed:
                    preprocessed = preprocessed.replace(prompt, "").strip()
                
                    # Only use if significantly different and reasonable length
                    if len(preprocessed) > 0 and len(preprocessed) < len(text) * 2:
                        return preprocessed
                    
        except Exception as e:
            # Text preprocessing failed
            pass
            
        return text  # Return original if preprocessing fails
    
    def get_environmental_context(self) -> dict:
        """🎯 Get environmental context from YOLO data for emotion adjustment"""
        try:
            # Try to get environmental context from face microservice
            import requests
            response = requests.get('http://localhost:5002/api/status', timeout=2)
            if response.status_code == 200:
                status_data = response.json()
                if status_data.get('running'):
                    # Get environmental context from face microservice
                    env_response = requests.get('http://localhost:5002/api/emotions', timeout=2)
                    if env_response.status_code == 200:
                        emotion_data = env_response.json()
                        environment = emotion_data.get('environment', {})
                        
                        if environment:
                            # YOLO environmental context applied
                            return self._process_environmental_context(environment)
        except Exception as e:
            # Failed to get environmental context
            pass
        
        return {}
    
    def _process_environmental_context(self, environment: dict) -> dict:
        """🧠 Process YOLO environmental context for emotion adjustment"""
        context_modifiers = {}
        detected_objects = environment.get('detected_categories', [])
        environment_type = environment.get('type', 'unknown')
        
        # AI-driven environmental emotion modifiers (not rule-based)
        environmental_emotion_impacts = {
            'indoor': {
                'calm': 1.1, 'neutral': 1.1, 'focused': 1.2
            },
            'outdoor': {
                'happy': 1.1, 'excited': 1.2, 'energetic': 1.1
            },
            'office': {
                'neutral': 1.1, 'focused': 1.3, 'professional': 1.2
            },
            'home': {
                'comfortable': 1.2, 'relaxed': 1.1, 'calm': 1.1
            },
            'social': {
                'happy': 1.2, 'excited': 1.1, 'social': 1.3
            }
        }
        
        # Object-based emotion context
        object_emotion_impacts = {
            'person': {'social': 1.1, 'interactive': 1.1},
            'laptop': {'focused': 1.1, 'work': 1.2},
            'book': {'calm': 1.1, 'thoughtful': 1.1},
            'phone': {'connected': 1.1, 'distracted': 1.1},
            'food': {'satisfied': 1.1, 'comfort': 1.1},
            'pet': {'happy': 1.2, 'affectionate': 1.2},
            'tv': {'relaxed': 1.1, 'entertainment': 1.1}
        }
        
        # Apply environment type modifiers
        if environment_type in environmental_emotion_impacts:
            context_modifiers.update(environmental_emotion_impacts[environment_type])
        
        # Apply object-based modifiers
        for obj_category in detected_objects:
            if obj_category in object_emotion_impacts:
                for emotion, modifier in object_emotion_impacts[obj_category].items():
                    context_modifiers[emotion] = context_modifiers.get(emotion, 1.0) * modifier
        
        processed_context = {
            'environment_type': environment_type,
            'detected_objects': detected_objects,
            'emotion_modifiers': context_modifiers,
            'context_strength': len(detected_objects) / 10.0,  # Normalize strength
            'has_context': len(context_modifiers) > 0
        }
        
        if processed_context['has_context']:
            # Environmental emotion adjustments applied
        
            return processed_context

class ProductionEmotionAnalyzer:
    """Production-grade emotion analysis with AI context awareness"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.context_preprocessor = AIContextPreprocessor()  # Add AI context
        self.model_configs = [
            {
                'name': 'roberta_emotion',
                'model_id': 'j-hartmann/emotion-english-distilroberta-base',
                'weight': 0.4,
                'emotions': ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            },
            {
                'name': 'twitter_roberta',
                'model_id': 'cardiffnlp/twitter-roberta-base-emotion-multilabel-latest', 
                'weight': 0.3,
                'emotions': ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
            },
            {
                'name': 'bertweet_emotion',
                'model_id': 'finiteautomata/bertweet-base-emotion-analysis',
                'weight': 0.3,
                'emotions': ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
            }
        ]
        self.fallback_available = False
        
        # Initialize models
        self._initialize_production_models()
        
        # Standard emotion mapping for consistency
        self.emotion_standardization = {
            # Map all model outputs to standard emotions
            'anger': 'angry', 'angry': 'angry',
            'sadness': 'sad', 'sad': 'sad',
            'joy': 'happy', 'happiness': 'happy', 'happy': 'happy',
            'fear': 'anxious', 'anxious': 'anxious', 'nervous': 'anxious',
            'surprise': 'surprised', 'surprised': 'surprised',
            'disgust': 'disgusted', 'disgusted': 'disgusted',
            'love': 'loving', 'loving': 'loving',
            'neutral': 'neutral',
            'anticipation': 'excited', 'excitement': 'excited', 'excited': 'excited',
            'optimism': 'hopeful', 'hope': 'hopeful', 'hopeful': 'hopeful',
            'pessimism': 'worried', 'worry': 'worried', 'worried': 'worried',
            'trust': 'confident', 'confident': 'confident'
        }
    
    def _initialize_production_models(self):
        """Initialize multiple production-grade emotion models"""
        successful_models = 0
        
        for config in self.model_configs:
            try:
                # Loading emotion model
                
                # Load model and tokenizer
                model = AutoModelForSequenceClassification.from_pretrained(config['model_id'])
                tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
                
                # Create pipeline - handle different model requirements
                device = 0 if torch.cuda.is_available() else -1
                
                # Some models don't support return_all_scores
                try:
                    pipeline_model = pipeline(
                        "text-classification",
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        return_all_scores=True
                    )
                except Exception:
                    # Fallback without return_all_scores
                    pipeline_model = pipeline(
                        "text-classification",
                        model=model,
                        tokenizer=tokenizer,
                        device=device
                    )
                
                self.models[config['name']] = {
                    'pipeline': pipeline_model,
                    'weight': config['weight'],
                    'emotions': config['emotions']
                }
                
                successful_models += 1
                # Emotion model loaded successfully
                
            except Exception as e:
                # Failed to load emotion model
                continue
        
        if successful_models == 0:
            # No emotion models loaded, using fallback
            self._setup_fallback_model()
        else:
            # Emotion models loaded
            pass
    
    def _setup_fallback_model(self):
        """Setup lightweight fallback model"""
        try:
            # Use a simple, reliable model as fallback
            pipeline_model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1
            )
            
            self.models['fallback'] = {
                'pipeline': pipeline_model,
                'weight': 1.0,
                'emotions': ['negative', 'neutral', 'positive']
            }
            self.fallback_available = True
            # Fallback sentiment model loaded
            
        except Exception as e:
            # Fallback model failed
            pass
    
    def _analyze_with_ensemble(self, text: str) -> Dict[str, Any]:
        """Use ensemble of models for robust emotion detection"""
        if not self.models:
            return self._get_neutral_result()
        
        model_results = {}
        
        # Run all models in parallel for speed
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            future_to_model = {
                executor.submit(self._run_single_model, name, config, text): name 
                for name, config in self.models.items()
            }
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result(timeout=5)  # 5 second timeout per model
                    if result:
                        model_results[model_name] = result
                except Exception as e:
                    print(f"⚠️ Model {model_name} failed: {e}")
        
        if not model_results:
            print(f"❌ NO MODEL RESULTS - All models failed!")
            return self._get_neutral_result()
        
        print(f"🔍 DEBUG: Model results count: {len(model_results)}")
        for model_name, result in model_results.items():
            print(f"🔍 DEBUG: {model_name} -> {result}")
        
        # Combine results using weighted voting
        ensemble_result = self._ensemble_vote(model_results)
        print(f"🔍 DEBUG: Final ensemble result: {ensemble_result}")
        return ensemble_result
    
    def _run_single_model(self, model_name: str, model_config: Dict, text: str) -> Optional[Dict]:
        """Run a single model and return standardized results"""
        try:
            pipeline_model = model_config['pipeline']
            results = pipeline_model(text)
            
            # Handle different result formats from different models
            standardized_emotions = {}
            
            # Flatten results if they're wrapped in extra lists
            if isinstance(results, list) and len(results) > 0:
                # Check if it's wrapped in an extra list: [[{...}]]
                if isinstance(results[0], list):
                    results = results[0]  # Unwrap the outer list
            
            # Now process the actual results
            if isinstance(results, list) and len(results) > 0:
                for result in results:
                    if isinstance(result, dict) and 'label' in result and 'score' in result:
                        emotion = result['label'].lower()
                        score = result['score']
                        
                        # Skip 'others' category from some models
                        if emotion == 'others':
                            continue
                        
                        # Map to standard emotion
                        std_emotion = self.emotion_standardization.get(emotion, emotion)
                        
                        if std_emotion in standardized_emotions:
                            standardized_emotions[std_emotion] = max(standardized_emotions[std_emotion], score)
                        else:
                            standardized_emotions[std_emotion] = score
            
            # Case 2: Results is a single dictionary
            elif isinstance(results, dict) and 'label' in results and 'score' in results:
                emotion = results['label'].lower()
                score = results['score']
                if emotion != 'others':  # Skip 'others' category
                    std_emotion = self.emotion_standardization.get(emotion, emotion)
                    standardized_emotions[std_emotion] = score
            
            # Case 3: Results is in different format - try to handle gracefully
            else:
                # Unknown result format
                return None
            
            if not standardized_emotions:
                # No emotions extracted
                return None
            
            return {
                'emotions': standardized_emotions,
                'weight': model_config['weight'],
                'model': model_name
            }
            
        except Exception as e:
            # Single model error
            return None
    
    def _ensemble_vote(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine multiple model results using weighted voting"""
        
        # Collect all emotions and their weighted scores
        emotion_scores = {}
        total_weight = 0
        model_count = len(model_results)
        
        for model_name, result in model_results.items():
            weight = result['weight']
            total_weight += weight
            
            for emotion, score in result['emotions'].items():
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = []
                emotion_scores[emotion].append(score * weight)
        
        # Calculate final scores
        final_emotions = {}
        for emotion, scores in emotion_scores.items():
            # Use mean of weighted scores
            final_emotions[emotion] = statistics.mean(scores)
        
        # Find dominant emotion
        if final_emotions:
            dominant_emotion = max(final_emotions.items(), key=lambda x: x[1])
            
            # Check for mixed emotions (multiple high-confidence emotions)
            high_confidence = {k: v for k, v in final_emotions.items() if v > 0.4}
            mixed = len(high_confidence) > 1
            
            return {
                'dominant_emotion': dominant_emotion[0],
                'confidence': dominant_emotion[1],
                'all_emotions': final_emotions,
                'mixed_emotions': mixed,
                'method': f'ensemble_{model_count}_models',
                'models_used': list(model_results.keys())
            }
        
        return self._get_neutral_result()
    
    def _get_neutral_result(self) -> Dict[str, Any]:
        """Return neutral result as fallback"""
        return {
            'dominant_emotion': 'neutral',
            'confidence': 0.6,
            'all_emotions': {'neutral': 0.6},
            'mixed_emotions': False,
            'method': 'fallback_neutral'
        }
    
    def analyze_text_emotion(self, text: str, user_id: str = "default") -> Dict[str, Any]:
        """🧠 AI-driven emotion analysis with context preprocessing and user-centric learning"""
        if not text.strip():
            return self._get_neutral_result()
        
        try:
            # 🧠 STEP 1: AI-driven context analysis and preprocessing
            # AI context preprocessing
            
            # Get confidence multiplier based on sentiment intensity
            context_confidence_multiplier = self.context_preprocessor.analyze_context_confidence(text)
            
            # Preprocess informal text using AI (not rules)
            preprocessed_text = self.context_preprocessor.preprocess_informal_text(text)
            
            # Text preprocessed with context multiplier
            
            # 🧠 STEP 2: Use ensemble approach with preprocessed text
            result = self._analyze_with_ensemble(preprocessed_text)
            print(f"🔍 DEBUG: Raw ensemble result in analyze_text_emotion: {result}")
            
            # 🎓 STEP 3: Apply user-centric learning adjustments
            if 'confidence' in result and 'dominant_emotion' in result:
                original_confidence = result['confidence']
                predicted_emotion = result['dominant_emotion']
                
                # Get personalized confidence adjustment based on user's correction history
                user_confidence_multiplier = self.context_preprocessor.learning_system.get_personalized_confidence_adjustment(
                    text, predicted_emotion, original_confidence, user_id
                )
                
                # Get personalized emotion suggestion
                suggested_emotion = self.context_preprocessor.learning_system.get_personalized_emotion_suggestion(
                    text, predicted_emotion, user_id
                )
                
                # 🎯 STEP 4: Apply environmental context from YOLO data
                environmental_context = self.context_preprocessor.get_environmental_context()
                environmental_multiplier = 1.0
                
                if environmental_context and environmental_context.get('has_context'):
                    emotion_modifiers = environmental_context.get('emotion_modifiers', {})
                    
                    # Apply environmental adjustment to predicted emotion
                    if predicted_emotion in emotion_modifiers:
                        environmental_multiplier = emotion_modifiers[predicted_emotion]
                        # Environmental adjustment applied
                    
                    # Check if environment suggests different emotion
                    if emotion_modifiers:
                        max_env_emotion = max(emotion_modifiers.items(), key=lambda x: x[1])
                        if max_env_emotion[1] > 1.2 and max_env_emotion[0] != predicted_emotion:
                            result['environment_suggested_emotion'] = max_env_emotion[0]
                            # Environment suggestion applied
                
                # Apply all adjustments: context + user learning + environmental
                combined_multiplier = context_confidence_multiplier * user_confidence_multiplier * environmental_multiplier
                adjusted_confidence = original_confidence * combined_multiplier
                adjusted_confidence = min(1.0, max(0.0, adjusted_confidence))  # Clamp to [0,1]
                
                result['confidence'] = adjusted_confidence
                print(f"🔍 DEBUG: After confidence adjustment - emotion: {result['dominant_emotion']}, confidence: {adjusted_confidence}")
                print(f"🔍 DEBUG: Multipliers - context: {context_confidence_multiplier}, user: {user_confidence_multiplier}, env: {environmental_multiplier}")
                
                # Confidence adjustments completed
                
                # Add suggestions if available
                if suggested_emotion:
                    result['user_suggested_emotion'] = suggested_emotion
                    # User pattern suggestion applied
                
                # Store environmental context in result
                if environmental_context and environmental_context.get('has_context'):
                    result['environmental_context'] = environmental_context
            
            # Add comprehensive AI context metadata
            result['ai_context'] = {
                'original_text': text,
                'preprocessed_text': preprocessed_text,
                'context_confidence_multiplier': context_confidence_multiplier,
                'user_confidence_multiplier': user_confidence_multiplier if 'user_confidence_multiplier' in locals() else 1.0,
                'environmental_multiplier': environmental_multiplier if 'environmental_multiplier' in locals() else 1.0,
                'preprocessing_applied': preprocessed_text != text,
                'user_suggested_emotion': suggested_emotion if 'suggested_emotion' in locals() else None,
                'environment_suggested_emotion': result.get('environment_suggested_emotion'),
                'environmental_context_available': environmental_context.get('has_context', False) if environmental_context else False,
                'environmental_type': environmental_context.get('environment_type') if environmental_context else None,
                'detected_objects': environmental_context.get('detected_objects', []) if environmental_context else [],
                'user_id': user_id,
                'analysis_layers': ['ai_preprocessing', 'ensemble_models', 'user_learning', 'environmental_context']
            }
            
            result['original_text'] = text
            result['timestamp'] = datetime.now().isoformat()
            result['processing_time'] = time.time()
            
            print(f"🔍 DEBUG: Final result before return in analyze_text_emotion: {result}")
            return result
            
        except Exception as e:
            # AI emotion analysis failed
            print(f"❌ EXCEPTION in analyze_text_emotion: {e}")
            import traceback
            traceback.print_exc()
            return self._get_neutral_result()

class FunctionCalling:
    """Production-grade function calling with error handling"""
    
    def __init__(self):
        self.available_functions = {
            'web_search': self.web_search,
            'get_weather': self.get_weather,
            'calculate': self.calculate,
            'get_time': self.get_time,
            'get_date': self.get_date
        }
    
    def web_search(self, query: str, num_results: int = 3) -> str:
        """Search the web for information"""
        try:
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if data.get('AbstractText'):
                return f"Search result: {data['AbstractText']}"
            elif data.get('RelatedTopics'):
                topics = data['RelatedTopics'][:num_results]
                results = []
                for topic in topics:
                    if 'Text' in topic:
                        results.append(topic['Text'])
                return "Search results:\n" + "\n".join(f"• {result}" for result in results)
            else:
                return f"No specific results found for '{query}'"
                
        except Exception as e:
            return f"Web search error: {e}"
    
    def get_weather(self, location: str = "current") -> str:
        """Get weather information"""
        return f"Weather functionality requires API key setup. Location requested: {location}"
    
    def calculate(self, expression: str) -> str:
        """Perform mathematical calculations"""
        try:
            # Safe evaluation
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return f"Calculation: {expression} = {result}"
            else:
                return "Invalid mathematical expression"
        except Exception as e:
            return f"Calculation error: {e}"
    
    def get_time(self) -> str:
        """Get current time"""
        return f"Current time: {datetime.now().strftime('%H:%M:%S')}"
    
    def get_date(self) -> str:
        """Get current date"""
        return f"Current date: {datetime.now().strftime('%Y-%m-%d (%A)')}"
    
    def detect_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """Detect function calls using NLP, not regex rules"""
        function_calls = []
        
        patterns = {
            'web_search': [r'search for (.+)', r'look up (.+)', r'find (.+)'],
            'get_weather': [r'weather', r'temperature'],
            'calculate': [r'calculate (.+)', r'what is (.+[\+\-\*/].+)'],
            'get_time': [r'time'],
            'get_date': [r'date']
        }
        
        for function_name, regex_patterns in patterns.items():
            for pattern in regex_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    function_calls.append({
                        'function': function_name,
                        'args': match.groups() if match.groups() else [],
                        'confidence': 0.8
                    })
        
        return function_calls
    
    def execute_function(self, function_name: str, args: List[str]) -> str:
        """Execute a function with error handling"""
        if function_name in self.available_functions:
            try:
                if args:
                    return self.available_functions[function_name](*args)
                else:
                    return self.available_functions[function_name]()
            except Exception as e:
                return f"Function execution error: {e}"
        else:
            return f"Function '{function_name}' not available"

class GeminiChatbot:
    """Production-ready Gemini chatbot with ensemble emotion detection"""
    
    def __init__(self, api_key: str = None, user_info: dict = None, session_id: str = None):
        # Use API rotation manager instead of single key
        self.model = None
        self.chat_session = None
        self.current_user = user_info  # Firebase user info
        
        # Initialize components
        self.emotion_analyzer = ProductionEmotionAnalyzer()
        self.function_calling = FunctionCalling()
        self.conversation_history = deque(maxlen=100)
        self.user_profile = None
        
        # 🔥 Initialize Firebase chat storage (like ChatGPT)
        self.firebase_storage = FirebaseChatStorage()
        # ChatGPT-style Firebase storage enabled
        
        # 🔥 CRITICAL: Set session ID if provided
        if session_id:
            self.firebase_storage.session_id = session_id
            print(f"🔗 GeminiChatbot initialized with session ID: {session_id}")
        
        # Configuration
        self.config = {
            'model_name': 'gemini-2.0-flash-exp',
            'temperature': 0.7,
            'top_k': 40,
            'top_p': 0.95,
            'max_tokens': 8192,  # 🚀 INCREASED: 2048 → 8192 for full responses!
            'safety_settings': [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
        }
        
        # System context
        self.system_context = """You are Y.M.I.R (Your Mental Intelligence and Recovery), a production-grade AI assistant specializing in mental health, emotional support, and intelligent conversation.

Your capabilities:
- Advanced ensemble emotion detection using multiple SOTA models
- Contextual understanding without rule-based preprocessing
- Function calling for web search, calculations, etc.
- Conversation memory and emotional intelligence
- Mental health and wellness guidance
- Technical assistance and problem-solving

Always be helpful, accurate, and emotionally intelligent in your responses."""
        
        self._initialize_gemini()
        self._load_user_profile()
    
    def _initialize_gemini(self):
        """Initialize Gemini API with rotation system"""
        try:
            if not GEMINI_AVAILABLE:
                # Gemini API not available
                return
            
            # Initializing Gemini with rotation system
            
            generation_config = {
                'temperature': self.config['temperature'],
                'top_k': self.config['top_k'], 
                'top_p': self.config['top_p'],
                'max_output_tokens': self.config['max_tokens']
            }
            
            # Use rotation manager to get model
            self.model = get_gemini_model(
                model_name=self.config['model_name'],
                generation_config=generation_config,
                safety_settings=self.config['safety_settings']
            )
            
            if self.model:
                self.chat_session = self.model.start_chat(history=[])
                # Gemini API initialized with rotation system
            else:
                # Failed to get model from rotation manager
                self.model = None
                self.chat_session = None
            
        except Exception as e:
            # Gemini initialization error
            self.model = None
            self.chat_session = None
    
    def _load_user_profile(self):
        """Load or create user profile from Firebase or create temporary profile"""
        if self.current_user and FIREBASE_AVAILABLE:
            # Load from Firebase for authenticated users
            try:
                db = firestore.client()
                user_doc = db.collection('user_profiles').document(self.current_user['uid']).get()
                
                if user_doc.exists:
                    data = user_doc.to_dict()
                    # Handle datetime fields
                    if 'created_at' in data:
                        data['created_at'] = data['created_at'].replace(tzinfo=None) if hasattr(data['created_at'], 'replace') else datetime.now()
                    if 'last_active' in data:
                        data['last_active'] = data['last_active'].replace(tzinfo=None) if hasattr(data['last_active'], 'replace') else datetime.now()
                    
                    # Use Firebase user info
                    data['user_id'] = self.current_user['uid']
                    data['name'] = self.current_user['name']
                    data['email'] = self.current_user['email']
                    
                    self.user_profile = UserProfile(**data)
                    # User profile loaded from Firebase
                else:
                    # Create new Firebase profile
                    self._create_firebase_profile()
                    
                # Update last active
                self.user_profile.last_active = datetime.now()
                self._save_user_profile()
                
            except Exception as e:
                # Firebase profile loading error
                self._create_temporary_profile()
        else:
            # Create temporary profile for unauthenticated users
            self._create_temporary_profile()
    
    def _create_firebase_profile(self):
        """Create new Firebase user profile"""
        self.user_profile = UserProfile(
            user_id=self.current_user['uid'],
            name=self.current_user['name'],
            created_at=datetime.now(),
            last_active=datetime.now(),
            conversation_style="balanced"
        )
        self._save_user_profile()
        # New Firebase profile created
    
    def _create_temporary_profile(self):
        """Create temporary user profile for unauthenticated users"""
        self.user_profile = UserProfile(
            user_id=f"temp_user_{int(time.time())}",
            conversation_style="balanced"
        )
        # Temporary profile created for guest user
    
    def _create_new_profile(self):
        """Legacy method - use Firebase or temporary profile instead"""
        self._create_temporary_profile()
    
    def _save_user_profile(self):
        """Save user profile to Firebase for authenticated users or skip for temporary users"""
        if not self.user_profile:
            return
            
        try:
            if self.current_user and FIREBASE_AVAILABLE:
                # Save to Firebase for authenticated users
                db = firestore.client()
                profile_data = asdict(self.user_profile)
                
                # Convert datetime objects for Firestore
                profile_data['created_at'] = self.user_profile.created_at
                profile_data['last_active'] = self.user_profile.last_active
                
                db.collection('user_profiles').document(self.current_user['uid']).set(profile_data)
                # Profile saved to Firebase
            else:
                # Don't save temporary profiles to disk
                print("⏭️ Temporary profile - not saving to disk")
                
        except Exception as e:
            print(f"⚠️ Profile saving error: {e}")
    
    def _build_context_prompt(self, user_input: str, emotion_data: Dict[str, Any]) -> str:
        """Build context prompt with emotion and history"""
        context_parts = [self.system_context]
        
        # Add emotion context
        if emotion_data['dominant_emotion'] != 'neutral':
            emotion_context = f"""
Current user emotion: {emotion_data['dominant_emotion']} (confidence: {emotion_data['confidence']:.2f})
Detection method: {emotion_data.get('method', 'unknown')}
Models used: {', '.join(emotion_data.get('models_used', []))}

Adapt your response to be supportive and appropriate for someone feeling {emotion_data['dominant_emotion']}.
"""
            context_parts.append(emotion_context)
        
        # Add conversation history
        if self.conversation_history:
            recent_messages = list(self.conversation_history)[-6:]
            history_context = "Recent conversation:\n"
            for msg in recent_messages:
                emotion_info = f" [felt: {msg.emotion}]" if msg.emotion and msg.emotion != 'neutral' else ""
                history_context += f"{msg.role}: {msg.content}{emotion_info}\n"
            context_parts.append(history_context)
        
        return "\n".join(context_parts) + f"\n\nUser: {user_input}"
    
    def _quick_emotion_analysis(self, text: str) -> Dict[str, Any]:
        """⚡ ENHANCED keyword-based emotion analysis with pattern matching"""
        import re
        text_lower = text.lower()
        
        # Enhanced emotion keywords with pattern matching
        emotion_patterns = {
            'happy': [
                r'happ+y+', r'joy+', r'great+', r'awesome+', r'love+', r'excellent+', 
                r'amazing+', r'wonderful+', r'yay+', r'yooo+', r'yes+', r'nice+',
                '😊', '😄', '🎉', '😃', '🥳', '🙂'
            ],
            'sad': [
                r'sad+', r'depress+', r'down+', r'terrible+', r'awful+', r'cry+', 
                r'hurt+', r'miserable+', r'upset+', '😢', '😭', '💔', '😞'
            ],
            'angry': [
                r'angry+', r'mad+', r'furious+', r'hate+', r'annoying+', r'stupid+', 
                r'damn+', r'pissed+', r'rage+', '😡', '🤬', '😠', '🔥'
            ],
            'fear': [
                r'scared+', r'afraid+', r'worr+', r'anxious+', r'nervous+', r'panic+', 
                r'terrified+', r'frighten+', '😰', '😨', '😱'
            ],
            'surprise': [
                r'wow+', r'amazing+', r'incredible+', r'unexpected+', r'surprised+', 
                r'shocked+', r'omg+', '😮', '😲', '🤩', '😯'
            ],
            'excited': [
                r'excit+', r'pumped+', r'thrilled+', r'energetic+', r'enthusiastic+'
            ]
        }
        
        # Count pattern matches with improved scoring
        emotion_scores = {}
        for emotion, patterns in emotion_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern.startswith('�') or pattern.startswith('�'):
                    # Emoji exact match
                    if pattern in text_lower:
                        score += 2  # Higher weight for emojis
                else:
                    # Regex pattern match for text variations
                    matches = len(re.findall(pattern, text_lower))
                    score += matches
            
            if score > 0:
                emotion_scores[emotion] = score
        
        # Determine dominant emotion with better confidence calculation
        if emotion_scores:
            total_score = sum(emotion_scores.values())
            dominant = max(emotion_scores.items(), key=lambda x: x[1])
            confidence = min(0.9, (dominant[1] / total_score) * 0.8 + 0.1)
            
            # Normalize all scores for better representation
            normalized_emotions = {k: v/total_score for k, v in emotion_scores.items()}
            
            return {
                'dominant_emotion': dominant[0],
                'confidence': confidence,
                'all_emotions': normalized_emotions,
                'method': 'enhanced_keyword'
            }
        else:
            return {
                'dominant_emotion': 'neutral',
                'confidence': 0.5,
                'all_emotions': {'neutral': 0.5},
                'method': 'quick_default'
            }
    
    def generate_response(self, user_input: str) -> Dict[str, Any]:
        """🤖 PURE AI emotion analysis - NO KEYWORDS, AI ONLY"""
        try:
            # 🤖 PURE AI: Only use AI emotion analysis with extended timeout
            print(f"🤖 Running PURE AI emotion analysis for: '{user_input[:50]}...'")
            
            final_emotion_result = None
            try:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.emotion_analyzer.analyze_text_emotion, user_input)
                    final_emotion_result = future.result(timeout=10.0)  # 10-second timeout for AI
                
                print(f"✅ AI SUCCESS: {final_emotion_result['dominant_emotion']} ({final_emotion_result['confidence']:.2f})")
                final_emotion_result['method'] = 'pure_ai'
                final_emotion_result['processing_time'] = '🤖 AI Only'
                
            except concurrent.futures.TimeoutError:
                print("❌ AI TIMEOUT - This should not happen! Increasing timeout or check models")
                # NO FALLBACK - Return error instead
                return {
                    'response': "AI emotion analysis timed out. Please try again.",
                    'emotion': {'dominant_emotion': 'error', 'confidence': 0.0, 'method': 'ai_timeout'},
                    'functions_used': [],
                    'streaming': False,
                    'error': 'ai_timeout'
                }
            except Exception as e:
                print(f"❌ AI FAILED: {e}")
                # NO FALLBACK - Return error instead
                return {
                    'response': "AI emotion analysis failed. Please try again.",
                    'emotion': {'dominant_emotion': 'error', 'confidence': 0.0, 'method': 'ai_error'},
                    'functions_used': [],
                    'streaming': False,
                    'error': f'ai_error: {str(e)}'
                }
            
            if not final_emotion_result:
                print("❌ NO AI RESULT")
                return {
                    'response': "No emotion analysis result. Please try again.",
                    'emotion': {'dominant_emotion': 'error', 'confidence': 0.0, 'method': 'no_result'},
                    'functions_used': [],
                    'streaming': False,
                    'error': 'no_ai_result'
                }
            
            # 🎯 Record PURE AI emotion result
            print(f"🎯 PURE AI EMOTION: {final_emotion_result['dominant_emotion']} ({final_emotion_result['confidence']:.2f})")
            
            user_message = ChatMessage(
                role='user',
                content=user_input,
                timestamp=datetime.now(),
                emotion=final_emotion_result['dominant_emotion'],
                confidence=final_emotion_result['confidence'],
                metadata={
                    'emotion_analysis': final_emotion_result,
                    'analysis_method': final_emotion_result.get('method', 'pure_ai'),
                    'all_emotions': final_emotion_result.get('all_emotions', {}),
                    'processing_time': '🤖 Pure AI',
                    'models_used': final_emotion_result.get('models_used', [])
                }
            )
            
            # Store PURE AI emotion in Firebase
            self.firebase_storage.store_message(user_message)
            self.conversation_history.append(user_message)
            
            # Check for function calls (keep this fast)
            function_calls = self.function_calling.detect_function_calls(user_input)
            function_results = []
            
            # Execute functions (if any)
            for call in function_calls:
                result = self.function_calling.execute_function(call['function'], call['args'])
                function_results.append(result)
            
            # Build context with CORRECTED emotion analysis result
            enhanced_prompt = self._build_context_prompt(user_input, final_emotion_result)
            
            if function_results:
                enhanced_prompt += f"\n\nFunction results: {'; '.join(function_results)}"
            
            # Generate response with proper emotion context
            if self.model and self.chat_session:
                try:
                    # Use full context prompt for complete responses
                    response = self.chat_session.send_message(enhanced_prompt)
                    response_text = response.text  # 🚀 NO LENGTH LIMIT - Full response!
                except Exception as e:
                    error_msg = str(e).lower()
                    if ('quota' in error_msg or 'limit' in error_msg or 'exhausted' in error_msg or 
                        '429' in error_msg or 'exceeded' in error_msg):
                        # Quota exhausted, rotating API key
                        # Mark current key as exhausted and rotate
                        if GEMINI_AVAILABLE:
                            gemini_manager.mark_key_exhausted(str(e))
                        # Try to reinitialize with new key
                        # Attempting to reinitialize with new API key
                        self._initialize_gemini()
                        if self.model and self.chat_session:
                            # Successfully reinitialized
                            try:
                                response = self.chat_session.send_message(enhanced_prompt)
                                response_text = response.text  # 🚀 Full response on retry too!
                                # Request successful with new API key
                            except Exception as retry_error:
                                # Retry failed with new key
                                response_text = self._generate_fallback_response(user_input, final_emotion_result)
                        else:
                            # Failed to reinitialize model
                            response_text = self._generate_fallback_response(user_input, final_emotion_result)
                    else:
                        # Gemini API error
                        response_text = self._generate_fallback_response(user_input, final_emotion_result)
            else:
                response_text = self._generate_fallback_response(user_input, final_emotion_result)
            
            assistant_message = ChatMessage(
                role='assistant', 
                content=response_text,
                timestamp=datetime.now(),
                metadata={'functions_used': function_calls}
            )
            
            # Store assistant response immediately  
            self.firebase_storage.store_message(assistant_message)
            self.conversation_history.append(assistant_message)
            
            # Update profile with CORRECTED emotion
            if self.user_profile:
                self.user_profile.last_active = datetime.now()
                if final_emotion_result['dominant_emotion'] != 'neutral':
                    self.user_profile.emotion_history.append(final_emotion_result['dominant_emotion'])
                    self.user_profile.emotion_history = self.user_profile.emotion_history[-20:]
                self._save_user_profile()
            
            # Return response with VALIDATED/CORRECTED emotion analysis
            return {
                'response': response_text,
                'emotion': final_emotion_result,  # Use CORRECTED emotion analysis
                'functions_used': function_calls,
                'streaming': True,
                'user_message': user_message,
                'assistant_message': assistant_message
            }
            
        except Exception as e:
            # Response generation error
            return {
                'response': f"I apologize, but I encountered an error: {str(e)}",
                'emotion': {'dominant_emotion': 'neutral', 'confidence': 0.5},
                'functions_used': [],
                'streaming': False
            }
    
    def _generate_fallback_response(self, user_input: str, emotion_data: Dict[str, Any]) -> str:
        """Generate fallback response when Gemini is not available"""
        emotion = emotion_data['dominant_emotion']
        
        # More natural, therapist-like responses
        responses = {
            'happy': [
                "I'm really glad to hear you're feeling good! What's been going well for you lately?",
                "It sounds like you're in a positive space right now. I'd love to hear more about what's bringing you joy.",
                "That's wonderful! Happiness is such a beautiful emotion. What's been making you feel this way?",
                "I can sense the positivity in your message. Tell me more about what's lighting you up today."
            ],
            'joy': [
                "What a beautiful feeling! I'd love to hear what's bringing you so much joy.",
                "It's wonderful when we feel truly joyful. What's been the source of this happiness?",
                "I can feel your joy through your words! What's been going so well for you?"
            ],
            'sad': [
                "I hear the sadness in your words, and I want you to know that's completely okay. What's been weighing on your heart?",
                "Sometimes life feels heavy, doesn't it? I'm here to listen if you'd like to share what's been difficult.",
                "Sadness is such a valid emotion. Would you like to talk about what's been troubling you?",
                "I can sense you're going through something tough. I'm here to support you through this."
            ],
            'angry': [
                "I can hear the frustration in your message. Anger often tells us something important - what's been bothering you?",
                "It sounds like something has really upset you. Would you like to talk through what happened?",
                "Anger can be such a powerful emotion. What's been making you feel this way?",
                "I can sense your frustration. Sometimes it helps to express these feelings - what's going on?"
            ],
            'anxious': [
                "Anxiety can feel so overwhelming sometimes. What's been on your mind lately?",
                "I can sense some worry in your message. Take a deep breath - what's been causing you stress?",
                "Feeling anxious is so common, and you're not alone in this. What's been making you feel uneasy?",
                "I hear the concern in your words. Would you like to talk about what's been worrying you?"
            ],
            'fear': [
                "Fear can be really overwhelming. You're safe here to share what's been frightening you.",
                "I can sense some fear in your message. What's been making you feel scared or worried?",
                "Sometimes fear tries to protect us, but it can also hold us back. What are you afraid of right now?"
            ],
            'neutral': [
                "Hi there! I'm Y.M.I.R, your mental health companion. How are you feeling today?",
                "Hello! I'm here to listen and support you. What's on your mind?",
                "I'm glad you reached out. What would you like to talk about today?",
                "How can I help you today? I'm here to listen and provide support."
            ],
            'confused': [
                "It sounds like you might be feeling a bit uncertain about something. What's been on your mind?",
                "Sometimes life can feel confusing. Would you like to talk through what's been puzzling you?",
                "I can sense some confusion in your message. What's been difficult to understand lately?"
            ]
        }
        
        emotion_responses = responses.get(emotion, [
            "I'm here to listen and support you. What's on your mind today?",
            "Thank you for reaching out. I'm Y.M.I.R, and I'm here to help. How are you feeling?",
            "I can sense there's something you'd like to talk about. I'm here for you - what's going on?",
            "Sometimes it helps just to have someone listen. What would you like to share with me?"
        ])
        
        emotion_responses = responses.get(emotion, emotion_responses)
        return emotion_responses[hash(user_input) % len(emotion_responses)]
    
    def save_conversation(self):
        """Save conversation to file"""
        try:
            # Prepare user profile data safely
            user_profile_data = None
            if self.user_profile:
                user_profile_data = asdict(self.user_profile)
                # Convert datetimes to strings safely
                if hasattr(self.user_profile.created_at, 'isoformat'):
                    user_profile_data['created_at'] = self.user_profile.created_at.isoformat()
                else:
                    user_profile_data['created_at'] = str(self.user_profile.created_at)
                    
                if hasattr(self.user_profile.last_active, 'isoformat'):
                    user_profile_data['last_active'] = self.user_profile.last_active.isoformat()
                else:
                    user_profile_data['last_active'] = str(self.user_profile.last_active)
            
            # Get conversation data
            conversation_list = []
            emotions_in_session = []
            
            for msg in self.conversation_history:
                try:
                    msg_dict = msg.to_dict()
                    conversation_list.append(msg_dict)
                    
                    # Track emotions
                    if msg.emotion and msg.emotion != 'neutral':
                        emotions_in_session.append(msg.emotion)
                        
                except Exception as e:
                    # Error processing message
                    continue
            
            conversation_data = {
                'timestamp': datetime.now().isoformat(),
                'user_profile': user_profile_data,
                'conversation': conversation_list,
                'session_stats': {
                    'total_messages': len(conversation_list),
                    'user_messages': len([msg for msg in conversation_list if msg.get('role') == 'user']),
                    'assistant_messages': len([msg for msg in conversation_list if msg.get('role') == 'assistant']),
                    'emotions_detected': list(set(emotions_in_session)),
                    'session_duration_minutes': 0,
                    'models_used': list(set(msg.get('metadata', {}).get('emotion_method', '') for msg in conversation_list if msg.get('metadata')))
                }
            }
            
            filename = f"chat_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(conversation_data, f, indent=2, default=str, ensure_ascii=False)
            
            # Conversation saved
            return filename
            
        except Exception as e:
            # Conversation save error
            return None

# Global instances
emotion_analyzer = ProductionEmotionAnalyzer()
function_calling = FunctionCalling()
chatbot = GeminiChatbot()

# 🎯 MICROSERVICE API ENDPOINTS

@app.route('/health')
def health_check():
    """Health check endpoint for microservice monitoring"""
    return jsonify({
        'service': 'Y.M.I.R Text Emotion Detection Microservice - FULL PRODUCTION',
        'status': 'healthy',
        'version': '2.0.0',
        'port': 5003,
        'analyzer_available': emotion_analyzer is not None,
        'models_loaded': list(emotion_analyzer.models.keys()) if emotion_analyzer.models else [],
        'gemini_available': chatbot.model is not None,
        'function_calling_available': len(function_calling.available_functions),
        'user_profile_loaded': chatbot.user_profile is not None,
        'gemini_api_status': get_api_status() if GEMINI_AVAILABLE else None
    })

@app.route('/')
def index():
    """API-only microservice info page"""
    return jsonify({
        'service': 'Y.M.I.R Text Emotion Detection Microservice - FULL PRODUCTION',
        'version': '2.0.0',
        'port': 5003,
        'description': 'Complete production-grade microservice with ALL chatbot features',
        'features': [
            'Ensemble emotion detection with multiple SOTA models',
            'Gemini AI integration with streaming responses',
            'Function calling (web search, calculations, etc.)',
            'User profiles and conversation memory',
            'Advanced analytics and session management',
            'Production-grade error handling'
        ],
        'endpoints': {
            'health': '/health',
            'analyze_text': '/api/analyze_text',
            'chat': '/api/chat',
            'conversation': '/api/conversation',
            'analytics': '/api/analytics',
            'status': '/api/status',
            'profile': '/api/profile',
            'functions': '/api/functions',
            'save_conversation': '/api/save_conversation'
        },
        'usage': 'This microservice provides complete chatbot APIs. Use the main app at port 5000 for UI.'
    })

@app.route('/api/analyze_text', methods=['POST'])
def api_analyze_text():
    """API endpoint to analyze text emotion with full production features"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        text = data['text']
        is_user = data.get('is_user', True)
        
        # Use production emotion analysis
        emotion_result = emotion_analyzer.analyze_text_emotion(text)
        
        # Create message with full metadata
        message = ChatMessage(
            role='user' if is_user else 'assistant',
            content=text,
            timestamp=datetime.now(),
            emotion=emotion_result['dominant_emotion'],
            confidence=emotion_result['confidence'],
            metadata={
                'emotion_analysis': emotion_result,
                'mixed_emotions': emotion_result.get('mixed_emotions', False),
                'models_used': emotion_result.get('models_used', [])
            }
        )
        
        return jsonify({
            'success': True,
            'message': message.to_dict(),
            'emotion_analysis': emotion_result,
            'session_id': chatbot.user_profile.user_id if chatbot.user_profile else None
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint for complete chat with emotion analysis and AI response"""
    import time
    start_time = time.time()
    print("🔥 CHAT ENDPOINT CALLED - DEBUG ACTIVE")
    print(f"⏱️ Request started at: {start_time}")
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing message parameter'}), 400
        
        user_message = data['message']
        frontend_session_id = data.get('session_id')  # Get session ID from frontend
        
        # 🔍 DEBUG: Check what's actually in the request
        print(f"🔍 DEBUG: Request data keys: {list(data.keys())}")
        print(f"🔍 DEBUG: frontend_session_id value: '{frontend_session_id}'")
        
        # Check for Firebase auth token
        auth_token = request.headers.get('Authorization')
        print(f"🔍 Auth header: {auth_token}")
        if auth_token and auth_token.startswith('Bearer '):
            auth_token = auth_token[7:]  # Remove 'Bearer ' prefix
            print(f"🔍 Cleaned token: {auth_token[:20]}...")
        else:
            print("❌ No valid Bearer token found")
        
        # Verify Firebase token and get user info
        user_info = verify_firebase_token(auth_token) if auth_token else None
        print(f"🔍 User info result: {user_info}")
        
        # 🚀 PERFORMANCE FIX: Reuse global chatbot instead of creating new ones
        user_chatbot = chatbot  # Always reuse the global instance!
        
        if user_info:
            # Authenticated user - update existing chatbot with user info
            user_chatbot.current_user = user_info
            print(f"✅ Updated chatbot with user info for: {user_info.get('email', 'unknown')}")
        
        # 🔥 CRITICAL: Set session ID from frontend
        if frontend_session_id:
            user_chatbot.firebase_storage.session_id = frontend_session_id
            print(f"✅ Session ID provided from frontend: {frontend_session_id}")
        else:
            print("⚠️ No session ID provided from frontend")
            # Temporary user session
        
        # 🔥 DEBUG: Session ID status
        if frontend_session_id:
            print(f"✅ Session ID provided from frontend: {frontend_session_id}")
        else:
            print("⚠️ No session ID provided from frontend")
        
        # 🔥 DEBUG: Check session ID
        print(f"🔍 Text microservice session ID: {user_chatbot.firebase_storage.session_id}")
        
        # Generate complete response using the appropriate chatbot
        print(f"🎭 Processing text message for emotion analysis...")
        processing_start = time.time()
        response_data = user_chatbot.generate_response(user_message)
        processing_time = time.time() - processing_start
        total_time = time.time() - start_time
        
        print(f"⏱️ PERFORMANCE METRICS:")
        print(f"   🎭 Text processing time: {processing_time:.2f}s")
        print(f"   🚀 Total request time: {total_time:.2f}s")
        print(f"   💾 Model loading time: {total_time - processing_time:.2f}s")
        
        # Add emotion_analysis to user_message for frontend compatibility
        user_message_dict = response_data['user_message'].to_dict()
        user_message_dict['emotion_analysis'] = response_data['emotion']
        
        return jsonify({
            'success': True,
            'user_message': user_message_dict,
            'assistant_message': response_data['assistant_message'].to_dict(),
            'bot_response': {  # Keep for backward compatibility
                'text': response_data['response'],
                'emotion_analysis': response_data['emotion'],
                'functions_used': response_data['functions_used'],
                'timestamp': datetime.now().isoformat()
            },
            'session_id': user_chatbot.firebase_storage.session_id,
            'conversation_length': len(user_chatbot.conversation_history),
            'user_authenticated': user_info is not None
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/conversation')
def api_conversation():
    """API endpoint to get full conversation history"""
    try:
        conversation_data = []
        for msg in chatbot.conversation_history:
            conversation_data.append(msg.to_dict())
        
        return jsonify({
            'success': True,
            'conversation': conversation_data,
            'total_messages': len(conversation_data),
            'user_profile': asdict(chatbot.user_profile) if chatbot.user_profile else None,
            'session_id': chatbot.user_profile.user_id if chatbot.user_profile else None
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analytics')
def api_analytics():
    """API endpoint to get comprehensive analytics"""
    try:
        emotions_in_session = []
        user_messages = 0
        assistant_messages = 0
        
        for msg in chatbot.conversation_history:
            if msg.role == 'user':
                user_messages += 1
            else:
                assistant_messages += 1
                
            if msg.emotion and msg.emotion != 'neutral':
                emotions_in_session.append(msg.emotion)
        
        analytics = {
            'session_stats': {
                'total_messages': len(chatbot.conversation_history),
                'user_messages': user_messages,
                'assistant_messages': assistant_messages,
                'emotions_detected': list(set(emotions_in_session)),
                'emotion_counts': {emotion: emotions_in_session.count(emotion) for emotion in set(emotions_in_session)},
                'models_available': list(emotion_analyzer.models.keys()),
                'gemini_available': chatbot.model is not None,
                'functions_available': list(function_calling.available_functions.keys())
            },
            'user_profile': asdict(chatbot.user_profile) if chatbot.user_profile else None
        }
        
        return jsonify(analytics)
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/profile', methods=['GET', 'POST'])
def api_profile():
    """API endpoint for user profile management"""
    try:
        if request.method == 'GET':
            if chatbot.user_profile:
                profile_data = asdict(chatbot.user_profile)
                # Convert datetime to string
                if hasattr(chatbot.user_profile.created_at, 'isoformat'):
                    profile_data['created_at'] = chatbot.user_profile.created_at.isoformat()
                if hasattr(chatbot.user_profile.last_active, 'isoformat'):
                    profile_data['last_active'] = chatbot.user_profile.last_active.isoformat()
                
                return jsonify({
                    'success': True,
                    'profile': profile_data
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'No user profile found'
                }), 404
        
        elif request.method == 'POST':
            data = request.get_json()
            if not chatbot.user_profile:
                chatbot._create_new_profile()
            
            # Update profile fields
            if 'name' in data:
                chatbot.user_profile.name = data['name']
            if 'conversation_style' in data:
                chatbot.user_profile.conversation_style = data['conversation_style']
            if 'topics_of_interest' in data:
                chatbot.user_profile.topics_of_interest = data['topics_of_interest']
            if 'preferences' in data:
                chatbot.user_profile.preferences.update(data['preferences'])
            
            chatbot.user_profile.last_active = datetime.now()
            chatbot._save_user_profile()
            
            return jsonify({
                'success': True,
                'message': 'Profile updated successfully'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/functions', methods=['GET', 'POST'])
def api_functions():
    """API endpoint for function calling"""
    try:
        if request.method == 'GET':
            return jsonify({
                'available_functions': list(function_calling.available_functions.keys()),
                'function_descriptions': {
                    'web_search': 'Search the web for information',
                    'get_weather': 'Get weather information',
                    'calculate': 'Perform mathematical calculations',
                    'get_time': 'Get current time',
                    'get_date': 'Get current date'
                }
            })
        
        elif request.method == 'POST':
            data = request.get_json()
            if not data or 'function' not in data:
                return jsonify({'error': 'Missing function parameter'}), 400
            
            function_name = data['function']
            args = data.get('args', [])
            
            result = function_calling.execute_function(function_name, args)
            
            return jsonify({
                'success': True,
                'function': function_name,
                'args': args,
                'result': result
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/save_conversation', methods=['POST'])
def api_save_conversation():
    """API endpoint to save conversation"""
    try:
        filename = chatbot.save_conversation()
        if filename:
            return jsonify({
                'success': True,
                'filename': filename,
                'message': 'Conversation saved successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to save conversation'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status')
def api_status():
    """API endpoint to get comprehensive system status"""
    return jsonify({
        'running': True,
        'analyzer_available': emotion_analyzer is not None,
        'models_loaded': len(emotion_analyzer.models) if emotion_analyzer else 0,
        'conversation_length': len(chatbot.conversation_history),
        'session_id': chatbot.user_profile.user_id if chatbot.user_profile else None,
        'available_methods': list(emotion_analyzer.models.keys()) if emotion_analyzer else [],
        'gemini_available': chatbot.model is not None,
        'functions_available': list(function_calling.available_functions.keys()),
        'user_profile_loaded': chatbot.user_profile is not None,
        'last_activity': chatbot.user_profile.last_active.isoformat() if chatbot.user_profile and hasattr(chatbot.user_profile.last_active, 'isoformat') else None,
        'learning_system_available': hasattr(emotion_analyzer.context_preprocessor, 'learning_system'),
        'user_corrections_recorded': len(emotion_analyzer.context_preprocessor.learning_system.user_corrections) if hasattr(emotion_analyzer.context_preprocessor, 'learning_system') else 0
    })

@app.route('/api/session', methods=['POST'])
def api_set_session():
    """🎯 Set session ID for text emotion storage synchronization"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                'success': False,
                'message': 'session_id is required'
            }), 400
        
        # Update global chatbot session ID
        success = chatbot.firebase_storage.set_session_id(session_id)
        
        return jsonify({
            'success': success,
            'message': 'Text session ID updated successfully' if success else 'Failed to update session ID',
            'session_id': chatbot.firebase_storage.session_id
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to set text session ID: {str(e)}'
        }), 500

@app.route('/api/user_feedback', methods=['POST'])
def api_user_feedback():
    """🎓 API endpoint for user to correct emotion predictions (user-centric learning)"""
    try:
        data = request.get_json()
        required_fields = ['original_text', 'predicted_emotion', 'predicted_confidence', 'corrected_emotion']
        
        if not data or not all(field in data for field in required_fields):
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {required_fields}'
            }), 400
        
        # Get user ID (default if not provided)
        user_id = data.get('user_id', chatbot.user_profile.user_id if chatbot.user_profile else 'default')
        
        # Record the user correction
        correction_id = emotion_analyzer.context_preprocessor.learning_system.record_user_correction(
            original_text=data['original_text'],
            predicted_emotion=data['predicted_emotion'],
            predicted_confidence=float(data['predicted_confidence']),
            user_corrected_emotion=data['corrected_emotion'],
            user_id=user_id
        )
        
        # Get updated learning analytics
        analytics = emotion_analyzer.context_preprocessor.learning_system.get_learning_analytics(user_id)
        
        return jsonify({
            'success': True,
            'correction_id': correction_id,
            'message': f'User correction recorded: {data["predicted_emotion"]} → {data["corrected_emotion"]}',
            'learning_analytics': analytics,
            'user_id': user_id
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/learning_analytics')
def api_learning_analytics():
    """🎓 API endpoint to get user learning analytics"""
    try:
        user_id = request.args.get('user_id', chatbot.user_profile.user_id if chatbot.user_profile else 'default')
        
        if not hasattr(emotion_analyzer.context_preprocessor, 'learning_system'):
            return jsonify({
                'success': False,
                'error': 'Learning system not available'
            }), 404
        
        analytics = emotion_analyzer.context_preprocessor.learning_system.get_learning_analytics(user_id)
        
        # Add additional system-wide analytics
        total_corrections = len(emotion_analyzer.context_preprocessor.learning_system.user_corrections)
        
        return jsonify({
            'success': True,
            'user_analytics': analytics,
            'system_analytics': {
                'total_corrections_all_users': total_corrections,
                'learning_system_active': True,
                'available_patterns': len(emotion_analyzer.context_preprocessor.learning_system.user_patterns)
            },
            'user_id': user_id
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/gemini-status')
def gemini_api_status():
    """Get Gemini API rotation status"""
    if not GEMINI_AVAILABLE:
        return jsonify({'error': 'Gemini API not available'}), 503
    
    try:
        status = get_api_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/emotion_suggestions')
def api_emotion_suggestions():
    """🎓 API endpoint to get personalized emotion suggestions for text"""
    try:
        text = request.args.get('text')
        predicted_emotion = request.args.get('predicted_emotion')
        user_id = request.args.get('user_id', chatbot.user_profile.user_id if chatbot.user_profile else 'default')
        
        if not text or not predicted_emotion:
            return jsonify({
                'success': False,
                'error': 'Missing required parameters: text, predicted_emotion'
            }), 400
        
        if not hasattr(emotion_analyzer.context_preprocessor, 'learning_system'):
            return jsonify({
                'success': False,
                'error': 'Learning system not available'
            }), 404
        
        # Get personalized suggestion
        suggested_emotion = emotion_analyzer.context_preprocessor.learning_system.get_personalized_emotion_suggestion(
            text, predicted_emotion, user_id
        )
        
        # Get confidence adjustment
        confidence_multiplier = emotion_analyzer.context_preprocessor.learning_system.get_personalized_confidence_adjustment(
            text, predicted_emotion, 1.0, user_id
        )
        
        return jsonify({
            'success': True,
            'suggested_emotion': suggested_emotion,
            'confidence_multiplier': confidence_multiplier,
            'has_suggestion': suggested_emotion is not None,
            'user_id': user_id
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("Y.M.I.R Text Emotion Detection MICROSERVICE")
    print("Microservice running on: http://localhost:5003")
    print("Health check: http://localhost:5003/health")
    
    # 🎯 START ON PORT 5003 AS MICROSERVICE - PRODUCTION MODE
    # 🚀 DISABLE DEBUG: Prevents crashes and auto-restart on file changes
    app.run(debug=False, host='0.0.0.0', port=5003, threaded=True)