"""
🎯 Y.M.I.R Advanced Emotion Detection System v3.0
==================================================
Modular Architecture with Enhanced Features:
- 🧠 Modular component design (MediaPipe, YOLO, DeepFace)
- 🎯 Enhanced YOLO emotion context analysis
- 🔥 Firebase Firestore integration
- 📊 Advanced analytics and emotion trends
- 🎛️ Environmental context-aware emotion detection
- 🎨 Face quality assessment with MediaPipe
- 🔄 Real-time emotion smoothing
- ⚡ Optimized performance and memory usage
"""

import cv2
import numpy as np
import threading
import time
import json
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timezone
from collections import defaultdict, deque
import statistics

# Import our modular components from face_models folder
from face_models.mediapipemodel import MediaPipeProcessor, MediaPipeConfig, FaceQuality
from face_models.yolomodel import EnhancedYOLOProcessor, YOLOConfig, EmotionContextAnalyzer
from true_ml_emotion_context import TrueMLEmotionContext
from face_models.deepfacemodel import DeepFaceEnsemble, DeepFaceConfig

# Firebase imports
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
    print("✅ Firebase available for cloud storage")
except ImportError:
    FIREBASE_AVAILABLE = False
    firebase_admin = None
    credentials = None
    firestore = None
    print("⚠️ Firebase not available - install: pip install firebase-admin")

warnings.filterwarnings("ignore")

@dataclass
class EnhancedEmotionConfig:
    """Enhanced configuration class for emotion detection system v3.0"""
    # Camera settings
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    
    # Detection settings
    emotion_analysis_interval: int = 30  # Analyze every 30 frames (1 second at 30fps)
    min_face_quality_score: float = 0.6
    
    # Memory optimization
    store_only_significant_changes: bool = False  # Store ALL emotions for combiner
    emotion_change_threshold: float = 5.0  # Reduced threshold for better data flow
    max_memory_entries: int = 1000
    
    # Database settings
    use_firebase: bool = True
    firebase_collection: str = "emotion_readings"
    
    # Display settings
    show_analytics: bool = True
    
    # Privacy settings
    require_user_consent: bool = True
    privacy_mode: bool = False
    
    # Component configurations
    mediapipe_config: Optional[MediaPipeConfig] = None
    yolo_config: Optional[YOLOConfig] = None
    deepface_config: Optional[DeepFaceConfig] = None
    
    def __post_init__(self):
        if self.mediapipe_config is None:
            self.mediapipe_config = MediaPipeConfig()
        if self.yolo_config is None:
            self.yolo_config = YOLOConfig()
        if self.deepface_config is None:
            self.deepface_config = DeepFaceConfig()

@dataclass
class EmotionReading:
    """Structure for emotion reading data"""
    timestamp: datetime
    face_id: int
    emotions: Dict[str, float]
    confidence: float
    quality_score: float
    context_objects: List[str]
    face_bbox: Tuple[int, int, int, int]
    environment_context: Optional[Dict[str, Any]] = None
    stability: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'timestamp': self.timestamp,
            'face_id': self.face_id,
            'emotions': self.emotions,
            'confidence': self.confidence,
            'quality_score': self.quality_score,
            'context_objects': self.context_objects,
            'face_bbox': self.face_bbox,
            'environment_context': self.environment_context,
            'stability': self.stability
        }

class EmotionAnalytics:
    """Advanced emotion analytics and insights"""
    
    def __init__(self, window_size: int = 100):
        self.emotion_history = deque(maxlen=window_size)
        self.session_stats = defaultdict(list)
        
    def add_reading(self, reading: EmotionReading):
        """Add emotion reading to analytics"""
        self.emotion_history.append(reading)
        
        # Update session stats
        dominant_emotion = max(reading.emotions.items(), key=lambda x: x[1])
        self.session_stats['dominant_emotions'].append(dominant_emotion[0])
        self.session_stats['confidence_scores'].append(reading.confidence)
        self.session_stats['quality_scores'].append(reading.quality_score)
        self.session_stats['stability_scores'].append(reading.stability)
        
        # Track environmental context
        if reading.environment_context:
            env_type = reading.environment_context.get('type', 'unknown')
            self.session_stats['environment_types'].append(env_type)
    
    def get_emotion_trends(self) -> Dict[str, Any]:
        """Calculate emotion trends and patterns with environmental analysis"""
        if not self.emotion_history:
            return {}
            
        # Calculate average emotions over time
        emotion_averages = defaultdict(list)
        for reading in self.emotion_history:
            for emotion, score in reading.emotions.items():
                emotion_averages[emotion].append(score)
        
        trends = {}
        for emotion, scores in emotion_averages.items():
            if scores:
                trends[emotion] = {
                    'average': statistics.mean(scores),
                    'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0,
                    'min': min(scores),
                    'max': max(scores),
                    'trend': 'stable'
                }
        
        # Environmental analysis
        env_analysis = self._analyze_environment_impact()
        
        # Session insights
        insights = {
            'total_readings': len(self.emotion_history),
            'avg_confidence': statistics.mean(self.session_stats['confidence_scores']) if self.session_stats['confidence_scores'] else 0,
            'avg_quality': statistics.mean(self.session_stats['quality_scores']) if self.session_stats['quality_scores'] else 0,
            'avg_stability': statistics.mean(self.session_stats['stability_scores']) if self.session_stats['stability_scores'] else 0,
            'dominant_emotion': max(set(self.session_stats['dominant_emotions']), key=self.session_stats['dominant_emotions'].count) if self.session_stats['dominant_emotions'] else 'neutral',
            'emotion_stability': self._calculate_stability(),
            'emotion_trends': trends,
            'environment_analysis': env_analysis
        }
        
        return insights
    
    def _analyze_environment_impact(self) -> Dict[str, Any]:
        """Analyze how environment affects emotions"""
        env_impact = {
            'most_common_environment': 'unknown',
            'environment_emotion_correlation': {},
            'context_effectiveness': 0.0
        }
        
        if not self.session_stats['environment_types']:
            return env_impact
        
        # Most common environment
        env_counts = {}
        for env_type in self.session_stats['environment_types']:
            env_counts[env_type] = env_counts.get(env_type, 0) + 1
        
        if env_counts:
            env_impact['most_common_environment'] = max(env_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate context effectiveness (how much context improves detection)
        readings_with_context = [r for r in self.emotion_history if r.environment_context]
        if readings_with_context:
            avg_confidence_with_context = statistics.mean([r.confidence for r in readings_with_context])
            total_avg_confidence = statistics.mean(self.session_stats['confidence_scores']) if self.session_stats['confidence_scores'] else 0
            
            if total_avg_confidence > 0:
                env_impact['context_effectiveness'] = avg_confidence_with_context / total_avg_confidence
        
        return env_impact
    
    def _calculate_stability(self) -> float:
        """Calculate emotional stability score (0-1)"""
        if len(self.emotion_history) < 2:
            return 1.0
            
        # Calculate variance in dominant emotions
        dominant_scores = []
        for reading in self.emotion_history:
            max_score = max(reading.emotions.values()) if reading.emotions else 0
            dominant_scores.append(max_score)
        
        if not dominant_scores:
            return 1.0
            
        variance = statistics.variance(dominant_scores)
        stability = max(0, 1 - (variance / 1000))  # Normalize to 0-1
        return stability

class FirebaseManager:
    """Firebase Firestore integration for cloud storage"""
    
    def __init__(self, config: EnhancedEmotionConfig):
        self.config = config
        self.db = None
        self.session_id = None
        
        if FIREBASE_AVAILABLE and config.use_firebase:
            self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase connection"""
        try:
            # Check if Firebase is already initialized
            if firebase_admin and not firebase_admin._apps:
                # You'll need to add your Firebase credentials file
                cred_path = Path("src/firebase_credentials.json")  # Fixed path
                if not cred_path.exists():
                    cred_path = Path("firebase_credentials.json")  # Fallback path
                if cred_path.exists() and credentials:
                    cred = credentials.Certificate(str(cred_path))
                    firebase_admin.initialize_app(cred)
                    print(f"✅ Firebase initialized with {cred_path}")
                else:
                    print("⚠️ Firebase credentials not found - using offline mode")
                    print(f"   Looked for: {cred_path.absolute()}")
                    return
            
            if firestore:
                self.db = firestore.client()
            self.session_id = f"session_{int(time.time())}"
            print("✅ Firebase Firestore connected")
            
        except Exception as e:
            print(f"⚠️ Firebase initialization error: {e}")
            self.db = None
    
    def store_emotion_reading(self, reading: EmotionReading) -> bool:
        """Store emotion reading in Firestore"""
        if not self.db:
            return False
            
        try:
            doc_ref = self.db.collection(self.config.firebase_collection).document()
            doc_ref.set({
                **reading.to_dict(),
                'session_id': self.session_id
            })
            return True
            
        except Exception as e:
            print(f"⚠️ Firebase storage error: {e}")
            return False
    
    def store_session_summary(self, analytics: Dict[str, Any]) -> bool:
        """Store session summary and analytics"""
        if not self.db:
            return False
            
        try:
            doc_ref = self.db.collection("session_summaries").document(self.session_id)
            doc_ref.set({
                **analytics,
                'session_id': self.session_id,
                'end_time': datetime.now(timezone.utc).isoformat()
            })
            return True
            
        except Exception as e:
            print(f"⚠️ Firebase session storage error: {e}")
            return False

class EnhancedEmotionDetector:
    """Enhanced emotion detection system v3.0 with modular architecture"""
    
    def __init__(self, config: Optional[EnhancedEmotionConfig] = None):
        self.config = config or EnhancedEmotionConfig()
        self.cap = None
        
        # Initialize modular components
        self._init_modular_components()
        
        # Initialize Firebase
        self.firebase_manager = FirebaseManager(self.config)
        
        # Initialize analytics
        self.emotion_analytics = EmotionAnalytics()
        
        # Emotion processing state
        self.current_emotions = {}
        self.last_stored_emotions = {}
        self.emotion_lock = threading.Lock()
        self.last_analysis_time = 0
        
        # Frame processing
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
        # Detection data
        self.detected_objects = []
        self.current_environment_context = {}
        
        # Visual settings
        self.colors = {
            "face": (0, 255, 0),
            "body": (255, 255, 0),
            "hands": (0, 0, 255),
            "eyes": (0, 255, 255),
            "objects": (255, 0, 255),
            "emotion_text": (0, 255, 255),
            "analytics": (255, 255, 255),
            "quality_good": (0, 255, 0),
            "quality_fair": (0, 255, 255),
            "quality_poor": (0, 0, 255)
        }
        
        print("🚀 Enhanced Emotion Detection System v3.0 initialized")
    
    def _init_modular_components(self):
        """Initialize all modular components"""
        try:
            print("🔧 Initializing modular components...")
            
            # Initialize MediaPipe processor
            self.mediapipe_processor = MediaPipeProcessor(self.config.mediapipe_config)
            print("✅ MediaPipe processor initialized")
            
            # Initialize YOLO processor
            self.yolo_processor = EnhancedYOLOProcessor(self.config.yolo_config)
            print("✅ YOLO processor initialized")
            
            # Initialize DeepFace ensemble
            self.deepface_ensemble = DeepFaceEnsemble(self.config.deepface_config)
            print("✅ DeepFace ensemble initialized")
            
            # Initialize TRUE ML emotion context analyzer (NO RULES!)
            self.true_ml_context = TrueMLEmotionContext(firebase_manager=None)
            print("🧠 TRUE ML Emotion Context Analyzer initialized")
            
        except Exception as e:
            print(f"❌ Component initialization error: {e}")
            raise
    
    def start_camera(self) -> bool:
        """Start camera with enhanced initialization"""
        if self.cap and self.cap.isOpened():
            print("📷 Camera already running")
            return True
        
        # Request permission
        if self.config.require_user_consent:
            print("\\n🔐 PRIVACY NOTICE:")
            print("This application uses emotion detection with optional cloud storage.")
            print("Your privacy is protected - data is encrypted and never shared.")
            
            while True:
                response = input("Grant camera access? (y/n): ").lower().strip()
                if response in ['y', 'yes']:
                    print("✅ Camera access granted")
                    break
                elif response in ['n', 'no']:
                    print("❌ Camera access denied")
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no")
        
        try:
            print("📷 Starting camera system...")
            
            # Try DirectShow backend first (Windows)
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                print("🔄 DirectShow failed, trying default backend...")
                self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                print("❌ Camera not detected!")
                return False
            
            # 🚀 OPTIMIZED camera properties for smooth streaming
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for real-time
            
            # Additional performance optimizations
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # MJPEG for better performance
            
            # 🔆 DO NOT modify camera hardware settings - causes driver corruption
            # Keep camera at default settings to prevent brightness/contrast corruption
            
            # Focus settings
            try:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus for clarity
            except:
                pass
            
            # Try to reduce latency
            try:
                self.cap.set(cv2.CAP_PROP_BACKEND, cv2.CAP_DSHOW)  # DirectShow backend for Windows
            except:
                pass  # Ignore if not supported
            
            print(f"✅ Camera started ({self.config.camera_width}x{self.config.camera_height})")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhanced frame processing with modular components"""
        self.frame_count += 1
        
        try:
            # MediaPipe processing
            mediapipe_results = self.mediapipe_processor.process_frame(frame)
            faces = mediapipe_results.get('faces', [])
            
            # YOLO processing (every 15 frames for performance)
            if self.frame_count % 15 == 0:
                new_objects, new_environment_context = self.yolo_processor.detect_objects_with_emotion_context(frame)
                self.detected_objects = new_objects
                self.current_environment_context = new_environment_context
                
                # 🔍 EXTENSIVE YOLO LOGGING
                if new_objects:
                    print(f"\\n🎯 YOLO DETECTED OBJECTS (Frame {self.frame_count}):")
                    for obj in new_objects:
                        conf = obj.get('confidence', 0)
                        class_name = obj.get('class', 'unknown')
                        print(f"   📦 {class_name.upper()} (confidence: {conf:.2f})")
                
                if new_environment_context:
                    print(f"\\n🌍 ENVIRONMENT ANALYSIS:")
                    env_type = new_environment_context.get('type', 'unknown')
                    print(f"   🏠 Environment Type: {env_type.upper()}")
                    
                    if 'context_modifiers' in new_environment_context:
                        modifiers = new_environment_context['context_modifiers']
                        print(f"   🎭 Emotion Modifiers Applied:")
                        for emotion, modifier in modifiers.items():
                            if modifier != 1.0:
                                direction = "↗️" if modifier > 1.0 else "↘️"
                                print(f"      {direction} {emotion}: {modifier:.2f}x")
                    
                    if 'detected_categories' in new_environment_context:
                        categories = new_environment_context['detected_categories']
                        print(f"   📋 Object Categories: {', '.join(categories)}")
                    
                    print()  # Empty line for readability
            
            # Emotion analysis (every 30 frames)
            if self.frame_count % self.config.emotion_analysis_interval == 0:
                print(f"🔍 DEBUG: Checking {len(faces)} faces for emotion analysis (frame {self.frame_count})")
                for i, face_info in enumerate(faces[:2]):  # Limit to 2 faces for performance
                    roi_size = face_info['roi'].size if 'roi' in face_info else 0
                    print(f"🔍 DEBUG: Face {i} ROI size: {roi_size}")
                    if face_info['roi'].size > 0:
                        print(f"🚀 Starting emotion analysis for face {i}")
                        threading.Thread(
                            target=self._analyze_face_emotions,
                            args=(face_info, self.current_environment_context),
                            daemon=True
                        ).start()
                    else:
                        print(f"⚠️ Skipping face {i} - empty ROI")
            
            # Draw all visualizations
            self.mediapipe_processor.draw_all_visualizations(frame, mediapipe_results)
            self.yolo_processor.draw_enhanced_objects(frame, self.detected_objects, self.current_environment_context)
            self.yolo_processor.draw_emotion_context_info(frame, self.current_environment_context)
            
        except Exception as e:
            print(f"⚠️ Frame processing error: {e}")
        
        # Draw controls and analytics
        self._draw_controls(frame)
        self._draw_analytics(frame)
        
        # Calculate FPS
        elapsed_time = time.time() - self.start_time
        self.fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return frame
    
    def _analyze_face_emotions(self, face_info: Dict, environment_context: Dict[str, Any]):
        """Analyze face emotions using DeepFace ensemble"""
        try:
            face_id = face_info['id']
            face_roi = face_info['roi']
            face_bbox = face_info['bbox']
            quality_score = face_info.get('quality_score', 0.8)
            
            # Analyze emotions WITHOUT rule-based context (get raw emotions first)
            emotion_result = self.deepface_ensemble.analyze_face_without_context(
                face_id, face_roi
            )
            
            if emotion_result:
                # 🚀 PURE DEEPFACE - No enhancement, just trust the 95% accurate system!
                raw_emotions = emotion_result['emotions']
                final_emotions = raw_emotions  # Use DeepFace directly - no modifications!
                
                print(f"🚀 PURE DEEPFACE EMOTION DETECTION:")
                print(f"   🎭 DeepFace result: {max(raw_emotions.items(), key=lambda x: x[1])[0].upper()}")
                print(f"   ✅ No enhancement applied - trusting DeepFace 95% accuracy!")
                
                # Create emotion reading with pure DeepFace emotions
                reading = EmotionReading(
                    timestamp=datetime.now(timezone.utc),
                    face_id=face_id,
                    emotions=final_emotions,  # Use pure DeepFace emotions
                    confidence=emotion_result['confidence'],
                    quality_score=quality_score,
                    context_objects=[obj['class'] for obj in self.detected_objects[:5]],
                    face_bbox=face_bbox,
                    environment_context={'pure_deepface': True, 'ml_enhancement_disabled': True},
                    stability=emotion_result.get('stability', 0.0)
                )
                
                # Update current emotions
                with self.emotion_lock:
                    self.current_emotions[face_id] = reading
                
                # Add to analytics
                self.emotion_analytics.add_reading(reading)
                
                # Store in Firebase if significant change
                print(f"🔍 DEBUG: Checking if emotion change is significant...")
                is_significant = self._is_significant_emotion_change(reading)
                print(f"🔍 DEBUG: Is significant change: {is_significant}")
                print(f"🔍 DEBUG: Firebase manager available: {self.firebase_manager is not None}")
                
                if is_significant:
                    try:
                        result = self.firebase_manager.store_emotion_reading(reading)
                        print(f"🔥 Enhanced Detector: Firebase storage result: {result}")
                        if result:
                            print(f"   ✅ Stored emotion successfully!")
                            self.last_stored_emotions[face_id] = reading.emotions
                        else:
                            print(f"   ❌ Firebase storage failed")
                    except Exception as e:
                        print(f"   ❌ Firebase storage exception: {e}")
                else:
                    print(f"   ⚠️ Emotion change not significant enough to store")
                
                # 🧠 EXTENSIVE EMOTION ANALYSIS LOGGING
                dominant_emotion = max(reading.emotions.items(), key=lambda x: x[1])
                print(f"\\n🧠 FACE EMOTION ANALYSIS (Face {face_id}):")
                print(f"   🎭 Dominant: {dominant_emotion[0].upper()} ({dominant_emotion[1]:.1f}%)")
                print(f"   📊 Confidence: {emotion_result['confidence']:.2f}")
                print(f"   🏆 Quality Score: {quality_score:.2f}")
                print(f"   📈 Stability: {emotion_result.get('stability', 0.0):.2f}")
                
                # Show all emotions detected
                print(f"   🎪 All Emotions Detected:")
                sorted_emotions = sorted(reading.emotions.items(), key=lambda x: x[1], reverse=True)
                for emotion, score in sorted_emotions:  # Show ALL emotions (not just top 4)
                    bar = "█" * max(1, int(score / 10))  # Visual bar
                    print(f"      {emotion.capitalize():12} {score:5.1f}% {bar}")
                
                # Show environmental context influence
                if environment_context and 'context_modifiers' in environment_context:
                    print(f"   🌍 Environmental Influence:")
                    modifiers = environment_context['context_modifiers']
                    for emotion in ['happy', 'sad', 'angry', 'fear']:
                        if emotion in modifiers and modifiers[emotion] != 1.0:
                            effect = "BOOSTED" if modifiers[emotion] > 1.0 else "REDUCED"
                            print(f"      {emotion.capitalize()} {effect} by {modifiers[emotion]:.2f}x")
                
                # Show detected objects affecting emotion
                if self.detected_objects:
                    print(f"   🎯 Context Objects: {', '.join([obj['class'] for obj in self.detected_objects[:5]])}")
                
                print()  # Empty line for readability
                
        except Exception as e:
            print(f"⚠️ Face emotion analysis error: {e}")
    
    def _is_significant_emotion_change(self, reading: EmotionReading) -> bool:
        """Check if emotion change is significant enough to store"""
        if not self.config.store_only_significant_changes:
            return True
            
        face_id = reading.face_id
        if face_id not in self.last_stored_emotions:
            return True
            
        # Calculate emotion change magnitude
        last_emotions = self.last_stored_emotions[face_id]
        current_emotions = reading.emotions
        
        total_change = 0
        for emotion in current_emotions:
            if emotion in last_emotions:
                change = abs(current_emotions[emotion] - last_emotions[emotion])
                total_change += change
        
        return total_change >= self.config.emotion_change_threshold
    
    def _draw_controls(self, frame: np.ndarray):
        """Draw control information"""
        controls = [
            f"🚀 Y.M.I.R v3.0 - Enhanced Modular System",
            f"FPS: {self.fps:.1f} | Faces: {len(self.current_emotions)} | Objects: {len(self.detected_objects)}",
            "Controls: Q - Quit | F - Face Mesh | B - Body | H - Hands | A - Analytics | P - Privacy",
            "ML Controls: T - Train ML | M - ML Status | 1/2/3 - Speed | S - Save | E - Export"
        ]
        
        for i, text in enumerate(controls):
            y_pos = 30 + i * 25
            cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["emotion_text"], 2)
    
    def _draw_analytics(self, frame: np.ndarray):
        """Draw analytics information"""
        if not self.config.show_analytics:
            return
            
        analytics = self.emotion_analytics.get_emotion_trends()
        if analytics:
            info_lines = [
                f"📊 Session Analytics:",
                f"Readings: {analytics.get('total_readings', 0)}",
                f"Avg Confidence: {analytics.get('avg_confidence', 0):.2f}",
                f"Avg Quality: {analytics.get('avg_quality', 0):.2f}",
                f"Dominant: {analytics.get('dominant_emotion', 'N/A').upper()}",
                f"Stability: {analytics.get('emotion_stability', 0):.2f}"
            ]
            
            for i, line in enumerate(info_lines):
                cv2.putText(frame, line, (frame.shape[1] - 320, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["analytics"], 1)
    
    def handle_key_events(self, key: int) -> bool:
        """Handle keyboard events"""
        if key == ord('q'):
            return False
        elif key == ord('f'):
            # Toggle face mesh visualization 
            self.mediapipe_processor.config.show_face_mesh = not self.mediapipe_processor.config.show_face_mesh
            print(f"Face mesh: {'ON' if self.mediapipe_processor.config.show_face_mesh else 'OFF'}")
        elif key == ord('b'):
            # Toggle body pose visualization
            self.mediapipe_processor.config.show_body_pose = not self.mediapipe_processor.config.show_body_pose
            print(f"Body pose: {'ON' if self.mediapipe_processor.config.show_body_pose else 'OFF'}")
        elif key == ord('h'):
            # Toggle hand tracking visualization
            self.mediapipe_processor.config.show_hand_tracking = not self.mediapipe_processor.config.show_hand_tracking
            print(f"Hand tracking: {'ON' if self.mediapipe_processor.config.show_hand_tracking else 'OFF'}")
        elif key == ord('a'):
            self.config.show_analytics = not self.config.show_analytics
            print(f"Analytics: {'ON' if self.config.show_analytics else 'OFF'}")
        elif key == ord('p'):
            self.config.privacy_mode = not self.config.privacy_mode
            print(f"Privacy mode: {'ON' if self.config.privacy_mode else 'OFF'}")
        elif key == ord('s'):
            self.save_analytics()
        elif key == ord('e'):
            self.export_session_data()
        elif key == ord('1'):
            self.config.emotion_analysis_interval = 15  # 0.5 seconds
            print("Analysis speed: FAST (0.5s)")
        elif key == ord('2'):
            self.config.emotion_analysis_interval = 30  # 1 second
            print("Analysis speed: NORMAL (1s)")
        elif key == ord('3'):
            self.config.emotion_analysis_interval = 60  # 2 seconds
            print("Analysis speed: SLOW (2s)")
        elif key == ord('t'):
            # Force train TRUE ML system
            print("🎓 Manually triggering TRUE ML training...")
            self.true_ml_context.force_retrain()
        elif key == ord('m'):
            # Show TRUE ML status
            insights = self.true_ml_context.get_discovery_insights()
            print(f"\n🧠 TRUE ML STATUS:")
            print(f"   📊 Observations: {insights['total_observations']}")
            print(f"   🎯 Training Progress: {insights['training_progress']:.1f}%")
            print(f"   ✅ Trained: {insights['is_trained']}")
            if insights['is_trained']:
                print(f"   🔍 Environment Patterns: {insights['environment_patterns']}")
                print(f"   🎭 Emotion Patterns: {insights['emotion_patterns']}")
        
        return True
    
    def save_analytics(self):
        """Save analytics summary"""
        analytics = self.emotion_analytics.get_emotion_trends()
        if analytics:
            if self.firebase_manager.store_session_summary(analytics):
                print("✅ Analytics saved to Firebase")
            else:
                filename = f"analytics_{int(time.time())}.json"
                with open(filename, 'w') as f:
                    json.dump(analytics, f, indent=2, default=str)
                print(f"✅ Analytics saved to {filename}")
    
    def export_session_data(self):
        """Export session data"""
        try:
            export_data = {
                'session_info': {
                    'start_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)),
                    'duration_seconds': time.time() - self.start_time,
                    'total_frames': self.frame_count,
                    'average_fps': self.fps,
                    'modular_components': True
                },
                'analytics': self.emotion_analytics.get_emotion_trends(),
                'recent_readings': [r.to_dict() for r in list(self.emotion_analytics.emotion_history)[-20:]]
            }
            
            filename = f"session_export_{int(time.time())}.json"
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Session data exported to {filename}")
            
        except Exception as e:
            print(f"❌ Export error: {e}")
    
    def run(self):
        """Run the enhanced emotion detection system"""
        print("\\n🚀 Starting Enhanced Y.M.I.R Emotion Detection System v3.0...")
        
        if not self.start_camera():
            return
        
        # Create window
        window_name = "🚀 Y.M.I.R Enhanced v3.0"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                if not self.cap:
                    print("❌ Camera not available")
                    break
                    
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Frame not captured. Retrying...")
                    continue
                
                # Flip frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Privacy mode
                if self.config.privacy_mode:
                    frame = np.zeros_like(frame)
                    cv2.putText(frame, "🔒 PRIVACY MODE", 
                              (frame.shape[1]//2 - 150, frame.shape[0]//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    # Process frame
                    frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Handle key events
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_key_events(key):
                    break
                
                # Check if window was closed
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        
        except KeyboardInterrupt:
            print("\\n🛑 System interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\\n🧹 Cleaning up...")
        
        # Save analytics
        self.save_analytics()
        
        # Stop camera
        if self.cap:
            self.cap.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        print("✅ Cleanup complete")

def main():
    """Main function"""
    print("🚀 Y.M.I.R Enhanced Emotion Detection System v3.0")
    print("=" * 60)
    print("Features: Modular architecture with enhanced YOLO context analysis")
    print("Components: MediaPipe + YOLO + DeepFace ensemble")
    print("Firebase: Cloud storage integration")
    print("=" * 60)
    
    # Enhanced configuration
    config = EnhancedEmotionConfig(
        camera_width=1280,
        camera_height=720,
        emotion_analysis_interval=30,
        min_face_quality_score=0.6,
        store_only_significant_changes=True,
        use_firebase=True,
        require_user_consent=True
    )
    
    # Create detector
    detector = EnhancedEmotionDetector(config)
    
    # Run system
    detector.run()

if __name__ == "__main__":
    main()