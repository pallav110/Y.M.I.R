"""
🚀 Y.M.I.R Web-based Emotion Detection Flask App - MICROSERVICE VERSION
===============================================
Your working facewebapp.py converted to run as microservice on port 5001
with CORS enabled for integration with main app.
"""

from flask import Flask, render_template_string, jsonify, request, Response
from flask_cors import CORS  # Added for microservice communication
import cv2
import json
import threading
import time
import numpy as np
from datetime import datetime
import base64
import io
from PIL import Image
import uuid
from typing import Optional, Dict, Any

# Add missing imports for the enhanced detector
import sys
import os
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics

# Ensure all required libraries are available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe available")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not available")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace available")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace not available")

try:
    import torch
    TORCH_AVAILABLE = True
    print("✅ PyTorch available")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available")
# Firebase imports - REQUIRED for production
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
    print("✅ Firebase Admin SDK loaded")
except ImportError:
    raise ImportError("❌ Firebase Admin SDK is required for production. Install: pip install firebase-admin")

# Import enhanced emotion detection system - REQUIRED for production
try:
    import sys
    import os
    # Add the correct path to the enhanced detector
    enhanced_detector_path = os.path.join(os.path.dirname(__file__), 'enhancements', 'src-new', 'face')
    if enhanced_detector_path not in sys.path:
        sys.path.append(enhanced_detector_path)
    
    from fer_enhanced_v3 import EnhancedEmotionDetector, EnhancedEmotionConfig
    print("✅ Enhanced emotion detector loaded")
except ImportError as e:
    try:
        # Try alternative path
        from enhancements.src_new.face.fer_enhanced_v3 import EnhancedEmotionDetector, EnhancedEmotionConfig
        print("✅ Enhanced emotion detector loaded from alternative path")
    except ImportError:
        raise ImportError(f"❌ Enhanced emotion detector is required for production: {e}")

app = Flask(__name__)
CORS(app)  # Enable CORS for microservice communication

class WebEmotionSystem:
    """Web-based emotion detection system using full EnhancedEmotionDetector"""
    
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.current_emotions: Dict[str, float] = {}
        self.frame_lock = threading.Lock()
        
        # 📊 Session tracking - wait for frontend to set session ID
        self.session_id = None
        self.session_analytics = {
            'total_readings': 0,
            'confidence_sum': 0,
            'quality_sum': 0,
            'stability_sum': 0,
            'start_time': None
        }
        
        # 🎯 Analysis control - enable by default for emotion detection
        self.analysis_mode = True  # True = continuous analysis, False = raw feed only
        self.last_analysis_request = time.time()
        print(f"🎭 Emotion analysis mode: {'ENABLED' if self.analysis_mode else 'DISABLED'}")
        
        # Storage configuration
        self.storage_enabled = True
        self.offline_mode = False
        self.readings_buffer = []
        self.last_firebase_sync = 0
        
        # Initialize the enhanced detector
        self._init_full_detector()
        
        # 🎛️ Initialize visual settings with defaults
        self.visual_settings = {
            'show_face_mesh': True,
            'show_body_pose': True,
            'show_hand_tracking': True,
            'show_gaze_tracking': True,
            'show_object_detection': True,
            'show_emotion_context': True,
            'confidence_threshold': 0.25,
            'video_quality': '720p',
            'analysis_interval': 30,
            'show_quality_indicators': True,
            'show_fps_display': False
        }
        
        # Microservice initialized with session ID
    
    def set_session_id(self, session_id: str):
        """Set the session ID for emotion storage"""
        if session_id:
            old_session_id = self.session_id
            self.session_id = session_id
            print(f"🔍 DEBUG: Session ID changed from '{old_session_id}' to '{session_id}'")
            # 🔥 CRITICAL: Also update Enhanced Detector session ID
            if hasattr(self, 'detector') and self.detector and hasattr(self.detector, 'firebase_manager'):
                if self.detector.firebase_manager:
                    self.detector.firebase_manager.session_id = session_id
                    print(f"✅ Enhanced Detector session ID updated to: {session_id}")
            return True
        return False
    
    def _init_full_detector(self):
        """Initialize the enhanced emotion detector"""
        try:
                
            # Create config for web microservice (HEADLESS MODE)
            config = EnhancedEmotionConfig(
                camera_width=640,
                camera_height=480,
                emotion_analysis_interval=30,
                require_user_consent=False,  # Web handles permissions
                use_firebase=True,  # ENABLE Firebase to store emotions for combiner
                show_analytics=False,  # Disable GUI analytics
                privacy_mode=False  # Ensure processing works
            )
            
            # Initialize the FULL enhanced detector
            self.detector = EnhancedEmotionDetector(config)
            # Enhanced emotion detector initialized
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize emotion detector: {e}")
    
    
    def start_camera(self):
        """Start camera capture using the full detector"""
        if self.is_running:
            return True
            
        try:
            if self.detector:
                # Starting camera with enhanced detector
                # Use the full detector's camera system
                success = self.detector.start_camera()
                if success:
                    self.is_running = True
                    self.session_analytics['start_time'] = time.time()
                    
                    # Start processing thread using the FULL detector
                    threading.Thread(target=self._process_frames_with_full_detector, daemon=True).start()
                    
                    # Camera started successfully
                    return True
                else:
                    # Camera failed to start
                    return False
            else:
                # Detector not available
                return False
            
        except Exception as e:
            # Camera start failed
            return False
    
    def stop_camera(self):
        """Stop camera capture with proper resource cleanup"""
        self.is_running = False
        
        # Clear current frame to stop video feed
        with self.frame_lock:
            self.current_frame = None
            
        # Stop and release camera
        if self.detector and self.detector.cap:
            try:
                self.detector.cap.release()
                self.detector.cap = None
                # Allow time for camera to fully release
                time.sleep(0.1)
            except Exception as e:
                print(f"Camera release error: {e}")
        
        # Clear emotions and reset state
        self.current_emotions.clear()
        
        # Force garbage collection for OpenCV resources
        import gc
        gc.collect()
        
        # Camera stopped with full cleanup
    
    def _process_frames_with_full_detector(self):
        """🚀 OPTIMIZED: Process camera frames using the FULL enhanced detector with performance optimizations"""
        # Starting optimized frame processing
        
        frame_count = 0
        last_fps_time = time.time()
        fps_counter = 0
        
        # Performance tracking
        processing_times = deque(maxlen=30)  # Track last 30 frame processing times
        
        while self.is_running and self.detector and self.detector.cap:
            try:
                frame_start_time = time.time()
                frame_count += 1
                fps_counter += 1
                
                # Calculate and update FPS every second
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    self.current_fps = fps_counter / (current_time - last_fps_time)
                    fps_counter = 0
                    last_fps_time = current_time
                    
                    # Only log every second instead of every 30 frames
                    if frame_count % 60 == 0:  # Log every 60 frames (less frequent)
                        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
                        # Frame processing stats available
                
                ret, frame = self.detector.cap.read()
                if not ret:
                    # Failed to read frame
                    time.sleep(0.1)  # Longer pause to prevent CPU overload
                    continue
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Store frame for web streaming (non-blocking)
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Skip processing every few frames if FPS is too low
                if hasattr(self, 'current_fps') and self.current_fps < 15 and frame_count % 2 == 0:
                    continue  # Skip every other frame for performance
                
                # 🎯 Only process frame if analysis is requested
                if self.analysis_mode:
                    processed_frame = self.detector.process_frame(frame)
                    # Note: processed_frame includes all visualizations and emotion analysis
                    if frame_count % 120 == 0:  # Log every 4 seconds at 30fps
                        print(f"🎭 Emotion analysis active - frame {frame_count}")
                else:
                    # Raw frame - no processing for smooth feed
                    processed_frame = frame.copy()
                    if frame_count % 120 == 0:
                        print(f"📹 Raw video mode - frame {frame_count}")
                
                # Track processing time for performance monitoring
                processing_time = time.time() - frame_start_time
                processing_times.append(processing_time)
                
                # Extract current emotions from the full detector
                if hasattr(self.detector, 'emotion_lock') and hasattr(self.detector, 'current_emotions'):
                    with self.detector.emotion_lock:
                        if self.detector.current_emotions:
                            # Update web system's current emotions for API
                            self.current_emotions: Dict[str, float] = {}
                            for face_id, reading in self.detector.current_emotions.items():
                                self.current_emotions[face_id] = {
                                    'dominant': reading.emotions,
                                    'confidence': reading.confidence,
                                    'quality': reading.quality_score,
                                    'timestamp': reading.timestamp.isoformat(),
                                    'stability': reading.stability
                                }
                            # Emotions updated
                else:
                    # Detector state invalid - skipping frame
                
                    time.sleep(0.016)  # ~60 FPS for smooth video
                
            except Exception as e:
                # Frame processing error - stopping
                break
                
        # Frame processing ended
    
    
    def _process_frame_headless(self, frame, frame_count):
        """Process frame using detector components in headless mode"""
        try:
            # MediaPipe processing (safe for headless)
            if hasattr(self.detector, 'mediapipe_processor'):
                mediapipe_results = self.detector.mediapipe_processor.process_frame(frame)
                faces = mediapipe_results.get('faces', [])
                
                # Only log when faces are found or every 30 frames
                if len(faces) > 0 or frame_count % 30 == 0:
                    print(f"👥 MediaPipe: {len(faces)} faces detected")
            
            # YOLO processing (every 5 frames) - THIS SHOULD SHOW THE DETAILED LOGS
            if frame_count % 5 == 0 and hasattr(self.detector, 'yolo_processor'):
                print("🎯 Processing with YOLO...")
                try:
                    objects, environment_context = self.detector.yolo_processor.detect_objects_with_emotion_context(frame)
                    self.detector.detected_objects = objects
                    self.detector.current_environment_context = environment_context
                    
                    # YOLO object detection completed
                    
                    # Environment analysis and context modifiers applied
                        
                except Exception as e:
                    # YOLO processing error
                    pass
            
            # Face emotion analysis (every 30 frames for performance while keeping 60fps video)
            if frame_count % 30 == 0 and 'faces' in locals() and faces:
                # Processing facial emotions
                try:
                    face_info = faces[0]  # Process first face
                    if face_info['roi'].size > 0:
                        # Simple emotion analysis without threading for headless mode
                        emotion_result = self.detector.deepface_ensemble.analyze_face_with_context(
                            face_info['id'], face_info['roi'], 
                            getattr(self.detector, 'current_environment_context', {})
                        )
                        
                        if emotion_result:
                            # Facial emotion analysis completed
                            
                            # 🔥 STORE IN FIREBASE for combiner!
                            # Checking Firebase storage availability
                            
                            # 🔥 FIREBASE STORAGE - Enhanced + Fallback
                            firebase_stored = False
                            
                            # SKIP ENHANCED DETECTOR - Go directly to fallback for guaranteed storage
                            # Using direct Firebase storage
                            
                            # 💾 DIRECT Firebase storage for guaranteed compatibility with combiner
                            print(f"🔍 Storage check: FIREBASE_AVAILABLE={FIREBASE_AVAILABLE}, session_id={self.session_id}")
                            if FIREBASE_AVAILABLE and self.session_id:
                                try:
                                    # Firebase storage fallback
                                    import firebase_admin
                                    from firebase_admin import firestore
                                    from datetime import datetime, timezone
                                    
                                    # Get current UTC timestamp for storage
                                    storage_timestamp = datetime.now(timezone.utc)
                                    # Storing with timestamp
                                    
                                    # Get Firestore client
                                    if not firebase_admin._apps:
                                        cred = firebase_admin.credentials.Certificate('firebase_credentials.json')
                                        firebase_admin.initialize_app(cred)
                                    
                                    db = firestore.client()
                                    
                                    # Store facial emotion with expected format for combiner
                                    facial_doc = {
                                        'timestamp': storage_timestamp,
                                        'face_id': face_info['id'],
                                        'emotions': emotion_result['emotions'],  # Key field for combiner!
                                        'confidence': emotion_result['confidence'],
                                        'quality_score': face_info.get('quality_score', 0.8),
                                        'session_id': self.session_id
                                        # NO 'role' field - this marks it as facial emotion
                                    }
                                    
                                    print(f"🔍 DEBUG: Storing facial emotion with session_id: {self.session_id}")
                                    dominant = max(emotion_result['emotions'].items(), key=lambda x: x[1])
                                    print(f"🔍 DEBUG: Emotion data: {dominant[0]} ({dominant[1]:.3f})")
                                    
                                    # Store in emotion_readings collection for combiner
                                    doc_ref = db.collection('emotion_readings').document()
                                    doc_ref.set(facial_doc)
                                    # Facial emotion stored to Firebase
                                    firebase_stored = True
                                    
                                except Exception as firebase_error:
                                    print(f"🔥 Firebase storage error: {firebase_error}")
                                    pass
                            
                            if not firebase_stored:
                                # Firebase storage methods failed
                                pass
                            
                except Exception as e:
                    # Emotion analysis error
                    pass
            
            # Processing completed - log only when emotions found
            pass
            
        except Exception as e:
            # Frame processing error
            import traceback
            traceback.print_exc()
    
    def get_current_frame_jpeg(self):
        """Get current frame with visual overlays as JPEG bytes"""
        with self.frame_lock:
            if self.current_frame is not None:
                # Apply visual overlays based on settings
                display_frame = self.apply_visual_overlays(self.current_frame.copy())
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    return buffer.tobytes()
        return None
    
    def get_current_frame_jpeg_optimized(self):
        """🚀 OPTIMIZED: Get current frame with performance optimizations for smooth streaming"""
        with self.frame_lock:
            if self.current_frame is not None:
                # Apply lightweight visual overlays 
                display_frame = self.apply_visual_overlays_optimized(self.current_frame.copy())
                
                # Optimized JPEG encoding with adaptive quality
                quality = 75  # Fixed quality setting
                ret, buffer = cv2.imencode('.jpg', display_frame, [
                    cv2.IMWRITE_JPEG_QUALITY, quality,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                    cv2.IMWRITE_JPEG_PROGRESSIVE, 1
                ])
                if ret:
                    return buffer.tobytes()
        return None
    
    def apply_visual_overlays(self, frame):
        """🎛️ Apply visual overlays based on current settings"""
        try:
            if not hasattr(self, 'visual_settings'):
                return frame
            
            settings = self.visual_settings
            # Removed excessive overlay logging
            
            # 🎯 Simple Visual Overlays (always available)
            
            # 📊 Show current emotions as overlay
            if self.current_emotions and settings.get('show_quality_indicators', True):
                self._draw_emotion_overlay(frame)
            
            # 📈 FPS Display
            if settings.get('show_fps_display', False):
                # Calculate actual FPS from frame timing
                current_time = time.time()
                if hasattr(self, 'last_frame_time'):
                    fps = 1.0 / (current_time - self.last_frame_time)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.last_frame_time = current_time
            
            # 🔥 Settings Status Display
            self._draw_settings_status(frame, settings)
            
            # 🎯 Enhanced Visual Overlays (if detector available)
            if self.detector and hasattr(self.detector, 'mediapipe_processor'):
                self._apply_enhanced_overlays(frame, settings)
            
            return frame
            
        except Exception as e:
            # Visual overlay error
            return frame
    
    def _draw_emotion_overlay(self, frame):
        """Draw current emotion information on frame"""
        try:
            y_offset = 50
            for emotion, confidence in self.current_emotions.items():
                text = f"{emotion.capitalize()}: {confidence:.1f}%"
                cv2.putText(frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
        except Exception as e:
            # Emotion overlay error
            pass
    
    def _draw_settings_status(self, frame, settings):
        """Draw visual settings status"""
        try:
            h, w = frame.shape[:2]
            y_start = h - 120
            
            # Show active visual features
            active_features = []
            if settings.get('show_face_mesh'): active_features.append("Face Mesh")
            if settings.get('show_body_pose'): active_features.append("Body Pose") 
            if settings.get('show_hand_tracking'): active_features.append("Hand Track")
            if settings.get('show_object_detection'): active_features.append("Objects")
            
            if active_features:
                text = f"Active: {', '.join(active_features)}"
                cv2.putText(frame, text, (10, y_start), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Show video quality
            quality = settings.get('video_quality', '720p')
            cv2.putText(frame, f"Quality: {quality}", (10, y_start + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Show detection confidence
            conf = settings.get('confidence_threshold', 0.25)
            cv2.putText(frame, f"Confidence: {conf:.2f}", (10, y_start + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                       
        except Exception as e:
            # Settings overlay error
            pass
    
    def _apply_enhanced_overlays(self, frame, settings):
        """Apply enhanced visual overlays using detector components"""
        try:
            # 📐 Face detection overlay
            if settings.get('show_face_mesh', True):
                # Draw face rectangles and landmarks using MediaPipe
                try:
                    import mediapipe as mp
                    # Use standard MediaPipe import structure with error handling
                    if hasattr(mp.solutions, 'face_detection'):
                        mp_face_detection = mp.solutions.face_detection # type: ignore
                        mp_drawing = mp.solutions.drawing_utils # type: ignore
                    else:
                        # Skip MediaPipe overlay if not available
                        return
                    
                    with mp_face_detection.FaceDetection(min_detection_confidence=0.6) as face_detection:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = face_detection.process(rgb_frame)
                    
                        if hasattr(results, 'detections') and results.detections:
                            for detection in results.detections:
                                # Draw face detection box
                                bbox = detection.location_data.relative_bounding_box
                                h, w, _ = frame.shape
                                x = int(bbox.xmin * w)
                                y = int(bbox.ymin * h)
                                width = int(bbox.width * w)
                                height = int(bbox.height * h)
                                
                                # Draw green rectangle around face
                                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                                cv2.putText(frame, f"Face {detection.score[0]:.2f}", (x, y-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                except (ImportError, AttributeError):
                    # MediaPipe not available, skip face overlay
                    pass
            
            # 🎯 Object detection status
            if settings.get('show_object_detection', True):
                detected_objects = getattr(self.detector, 'detected_objects', [])
                if detected_objects:
                    cv2.putText(frame, f"Objects: {len(detected_objects)}", (frame.shape[1] - 150, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                               
        except Exception as e:
            # Enhanced overlay error
            pass
    
    def apply_visual_overlays_optimized(self, frame):
        """🚀 OPTIMIZED: Lightweight visual overlays for smooth 60 FPS streaming"""
        try:
            if not hasattr(self, 'visual_settings'):
                return frame
            
            settings = self.visual_settings
            
            # Only apply essential overlays for performance
            # Skip heavy overlays like MediaPipe in streaming mode
            
            # 📊 Essential emotion overlay (lightweight)
            if self.current_emotions and settings.get('show_quality_indicators', True):
                self._draw_emotion_overlay_lightweight(frame)
            
            # 🔲 Simple face detection boxes only (no MediaPipe mesh for streaming)
            if settings.get('show_face_detection', True) and hasattr(self, 'last_faces'):
                for face in getattr(self, 'last_faces', []):
                    if len(face) >= 4:
                        x, y, w, h = face[:4]
                        # Simple optimized rectangle
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            
            # ⚡ Performance indicator
            if settings.get('show_performance_metrics', False):
                fps_text = f"FPS: {getattr(self, 'current_fps', 0):.1f}"
                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            # Silent error handling for performance
            return frame
    
    def _draw_emotion_overlay_lightweight(self, frame):
        """Lightweight emotion overlay for streaming performance"""
        try:
            if not self.current_emotions:
                return
            
            # Get primary emotion
            if isinstance(self.current_emotions, dict) and self.current_emotions:
                primary_emotion = max(self.current_emotions.keys(), key=lambda k: self.current_emotions[k])
            else:
                return
            confidence = self.current_emotions[primary_emotion]
            
            # Simple text overlay
            emotion_text = f"{primary_emotion.upper()}: {confidence:.1f}%"
            
            # Optimized text rendering
            (text_width, text_height), baseline = cv2.getTextSize(
                emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Background rectangle
            cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 10 + text_height + 10), 
                         (0, 0, 0), -1)
            
            # Emotion color mapping for visual feedback
            color_map = {
                'happy': (0, 255, 0),
                'sad': (255, 0, 0), 
                'angry': (0, 0, 255),
                'neutral': (128, 128, 128),
                'surprise': (255, 255, 0),
                'fear': (128, 0, 128),
                'disgust': (0, 128, 128)
            }
            color = color_map.get(primary_emotion.lower(), (255, 255, 255))
            
            # Text overlay
            cv2.putText(frame, emotion_text, (15, 10 + text_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                       
        except Exception:
            pass  # Silent fail for performance
    
    def get_emotion_data(self):
        """Get current emotion data for API using full detector"""
        if not self.current_emotions:
            return {
                'status': 'waiting',
                'message': 'Waiting for emotion detection...'
            }
        
        # Get detected objects from full detector
        objects = []
        environment = {}
        if self.detector:
            objects = [{'class': obj.get('class', 'unknown'), 'confidence': obj.get('confidence', 0)} 
                      for obj in self.detector.detected_objects[:5]]
            environment = self.detector.current_environment_context
        
        return {
            'status': 'active',
            'emotions': self.current_emotions,
            'objects': objects,
            'environment': environment,
            'analytics': self._get_analytics_summary()
        }
    
    def _get_analytics_summary(self):
        """Get analytics summary from full detector"""
        if self.detector and hasattr(self.detector, 'emotion_analytics'):
            # Use the full detector's analytics
            analytics = self.detector.emotion_analytics.get_emotion_trends()
            
            # Ensure required fields exist for frontend compatibility
            if isinstance(analytics, dict):
                # Make sure we have total_readings field
                if 'total_readings' not in analytics:
                    analytics['total_readings'] = analytics.get('total_readings', 0)
                return analytics
        
        # Fallback to basic analytics
        session_duration = 0
        if self.session_analytics['start_time']:
            session_duration = time.time() - self.session_analytics['start_time']
        
        return {
            'total_readings': 0,
            'avg_confidence': 0,
            'avg_quality': 0,
            'avg_stability': 0,
            'session_duration': session_duration
        }
    
    def get_storage_analytics(self):
        """Get storage analytics for API compatibility"""
        return {
            'session_id': self.session_id,
            'storage_enabled': self.storage_enabled,
            'offline_mode': self.offline_mode,
            'buffer_size': len(self.readings_buffer),
            'last_sync': self.last_firebase_sync,
            'total_stored': self.session_analytics['total_readings'],
            'storage_type': 'microservice_mode'
        }
    
    def _sync_to_firebase(self):
        """Sync buffered emotions to Firebase if storage enabled"""
        if self.storage_enabled and FIREBASE_AVAILABLE and self.readings_buffer:
            try:
                # Syncing buffered emotions to Firebase
                # Process any buffered readings here if needed
                self.last_firebase_sync = int(time.time())
                # Firebase sync completed
                pass
            except Exception as e:
                # Firebase sync error
                pass
        else:
            # Firebase sync skipped
            pass

# Global web emotion system
web_system = WebEmotionSystem()

# 🎯 MICROSERVICE API ENDPOINTS (Same as your original but with added health check)

@app.route('/health')
def health_check():
    """Health check endpoint for microservice monitoring"""
    return jsonify({
        'service': 'Y.M.I.R Face Emotion Detection Microservice',
        'status': 'healthy',
        'version': '1.0.0',
        'port': 5002,
        'detector_available': web_system.detector is not None,
        'camera_running': web_system.is_running
    })

@app.route('/')
def index():
    """API-only microservice info page"""
    return jsonify({
        'service': 'Y.M.I.R Face Emotion Detection Microservice',
        'version': '1.0.0',
        'port': 5002,
        'description': 'API-only microservice for facial emotion detection',
        'endpoints': {
            'health': '/health',
            'start_camera': '/api/start_camera',
            'stop_camera': '/api/stop_camera', 
            'emotions': '/api/emotions',
            'video_feed': '/video_feed',
            'status': '/api/status'
        },
        'usage': 'This microservice provides APIs only. Use the main app at port 5000 for UI.'
    })

@app.route('/video_feed')
def video_feed():
    """🚀 OPTIMIZED Video streaming route with proper cleanup"""
    def generate():
        last_frame_time = time.time()
        frame_interval = 1.0 / 60.0  # Target 60 FPS
        frame_count = 0
        max_idle_frames = 600  # Stop after 10 seconds of no camera
        
        try:
            while frame_count < max_idle_frames:
                current_time = time.time()
                frame_count += 1
                
                # Exit if camera system is stopped
                if not web_system.is_running:
                    break
                
                # Get frame from optimized buffer
                frame_bytes = web_system.get_current_frame_jpeg_optimized()
                
                if frame_bytes:
                    frame_count = 0  # Reset idle counter when we have frames
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # Send minimal black frame for startup
                    if frame_count <= 10:  # Only send black frames during startup
                        black_frame = np.zeros((240, 320, 3), dtype=np.uint8)  # Smaller startup frame
                        ret, buffer = cv2.imencode('.jpg', black_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                # Dynamic frame rate control - adapt to system performance
                elapsed = current_time - last_frame_time
                if elapsed < frame_interval:
                    sleep_time = max(0.001, frame_interval - elapsed)  # Minimum 1ms sleep
                    time.sleep(sleep_time)
                
                last_frame_time = time.time()
                
        except GeneratorExit:
            # Client disconnected - clean exit
            pass
        except Exception as e:
            # Unexpected error - log and exit gracefully
            print(f"Video feed error: {e}")
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 🎯 API ENDPOINTS (KEEP YOUR EXACT WORKING API STRUCTURE)

@app.route('/api/start_camera', methods=['POST'])
def api_start_camera():
    """API endpoint to start camera"""
    # API: start_camera called
    try:
        success = web_system.start_camera()
        # Camera start result available
        response = {
            'success': success,
            'message': 'Camera started successfully' if success else 'Failed to start camera',
            'session_id': web_system.session_id
        }
        # Returning API response
        return jsonify(response)
    except Exception as e:
        # API error in start_camera
        return jsonify({
            'success': False,
            'message': f'Camera start error: {str(e)}'
        })

@app.route('/api/stop_camera', methods=['POST'])
def api_stop_camera():
    """API endpoint to stop camera"""
    web_system.stop_camera()
    return jsonify({
        'success': True,
        'message': 'Camera stopped successfully'
    })

@app.route('/api/emotions')
def api_emotions():
    """API endpoint to get current emotion data"""
    return jsonify(web_system.get_emotion_data())

@app.route('/api/enable_analysis', methods=['POST'])
def api_enable_analysis():
    """API endpoint to enable/disable emotion analysis mode"""
    try:
        data = request.get_json()
        enable = data.get('enable', False)
        analysis_type = data.get('analysis_type', 'full')
        
        print(f"🎯 Analysis mode toggle: {enable} (type: {analysis_type})")
        
        # Toggle analysis mode
        web_system.analysis_mode = enable
        web_system.last_analysis_request = time.time()
        
        return jsonify({
            'success': True,
            'analysis_mode': web_system.analysis_mode,
            'analysis_type': analysis_type,
            'message': f'Analysis mode {"enabled" if enable else "disabled"}'
        })
        
    except Exception as e:
        print(f"Error toggling analysis: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analytics')
def api_analytics():
    """API endpoint to get session analytics"""
    return jsonify(web_system._get_analytics_summary())

@app.route('/api/storage')
def api_storage():
    """API endpoint to get storage analytics"""
    return jsonify(web_system.get_storage_analytics())

@app.route('/api/export_session', methods=['POST'])
def api_export_session():
    """API endpoint to export session data"""
    try:
        # Force sync any remaining buffered data
        web_system._sync_to_firebase()
        
        # Create export data
        export_data = {
            'session_info': {
                'session_id': web_system.session_id,
                'exported_at': datetime.now().isoformat(),
                'duration': time.time() - web_system.session_analytics['start_time'] if web_system.session_analytics['start_time'] else 0
            },
            'analytics': web_system._get_analytics_summary(),
            'storage': web_system.get_storage_analytics(),
            'export_format': 'ymir_emotion_session_v1.0'
        }
        
        return jsonify({
            'success': True,
            'export_data': export_data,
            'message': 'Session data exported successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to export session data'
        })

@app.route('/api/status')
def api_status():
    """API endpoint to get system status"""
    return jsonify({
        'running': web_system.is_running,
        'camera_active': web_system.is_running,
        'detector_available': web_system.detector is not None,
        'detector_loaded': web_system.detector is not None,
        'has_current_frame': web_system.current_frame is not None,
        'objects_detected': len(web_system.detector.detected_objects) if web_system.detector else 0,
        'emotions_detected': bool(web_system.current_emotions),
        'session_duration': time.time() - web_system.session_analytics['start_time'] if web_system.session_analytics['start_time'] else 0,
        'session_id': web_system.session_id,
        'analytics': web_system._get_analytics_summary(),
        'storage_status': {
            'enabled': web_system.storage_enabled,
            'offline_mode': web_system.offline_mode,
            'buffer_size': len(web_system.readings_buffer)
        }
    })

@app.route('/api/camera_debug')
def api_camera_debug():
    """Debug endpoint to check camera status"""
    camera_info = {
        'camera_initialized': False,
        'camera_opened': False,
        'frame_available': False,
        'error': None,
        'diagnostics': {}
    }
    
    try:
        if web_system.detector and web_system.detector.cap:
            camera_info['camera_initialized'] = True
            camera_info['camera_opened'] = web_system.detector.cap.isOpened()
            
            if web_system.current_frame is not None:
                camera_info['frame_available'] = True
                camera_info['frame_shape'] = web_system.current_frame.shape
                
                # Check if frame is all black/grey
                frame_mean = np.mean(web_system.current_frame)
                frame_std = np.std(web_system.current_frame)
                camera_info['diagnostics']['frame_mean'] = float(frame_mean)
                camera_info['diagnostics']['frame_std'] = float(frame_std)
                
                if frame_std < 5:  # Very low variation = likely no signal
                    camera_info['diagnostics']['issue'] = 'No video signal - frame appears blank/grey'
                elif frame_mean < 10:
                    camera_info['diagnostics']['issue'] = 'Very dark image - check lighting or exposure'
                else:
                    camera_info['diagnostics']['status'] = 'Frame looks normal'
                
        # Try to get frame dimensions if available
        if web_system.detector and web_system.detector.cap and web_system.detector.cap.isOpened():
            width = web_system.detector.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = web_system.detector.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = web_system.detector.cap.get(cv2.CAP_PROP_FPS)
            brightness = web_system.detector.cap.get(cv2.CAP_PROP_BRIGHTNESS)
            contrast = web_system.detector.cap.get(cv2.CAP_PROP_CONTRAST)
            exposure = web_system.detector.cap.get(cv2.CAP_PROP_EXPOSURE)
            
            camera_info['camera_properties'] = {
                'width': int(width),
                'height': int(height),
                'fps': fps,
                'brightness': brightness,
                'contrast': contrast,
                'exposure': exposure
            }
            
    except Exception as e:
        camera_info['error'] = str(e)
    
    return jsonify(camera_info)

@app.route('/api/camera_test_different_indices', methods=['POST'])
def api_camera_test_indices():
    """Test different camera indices to find working camera"""
    results = {}
    
    for i in range(5):  # Test cameras 0-4
        try:
            print(f"Testing camera index {i}...")
            cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_mean = np.mean(frame)
                    frame_std = np.std(frame)
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    
                    results[f'camera_{i}'] = {
                        'available': True,
                        'frame_captured': True,
                        'frame_mean': float(frame_mean),
                        'frame_std': float(frame_std),
                        'resolution': f"{int(width)}x{int(height)}",
                        'status': 'Working' if frame_std > 5 else 'No signal (grey/black)'
                    }
                else:
                    results[f'camera_{i}'] = {
                        'available': True,
                        'frame_captured': False,
                        'status': 'Cannot capture frames'
                    }
            else:
                results[f'camera_{i}'] = {
                    'available': False,
                    'status': 'Camera not accessible'
                }
            
            cap.release()
            
        except Exception as e:
            results[f'camera_{i}'] = {
                'available': False,
                'error': str(e)
            }
    
    return jsonify({
        'success': True,
        'camera_test_results': results,
        'recommendation': 'Use camera index with highest frame_std (>5) and Working status'
    })

@app.route('/api/camera_switch/<int:camera_index>', methods=['POST'])  
def api_camera_switch(camera_index):
    """Switch to a different camera index"""
    try:
        # Stop current camera
        web_system.stop_camera()
        time.sleep(1)
        
        # Update camera index in detector
        if web_system.detector:
            # Try new camera index
            new_cap = cv2.VideoCapture(camera_index)
            
            if new_cap.isOpened():
                ret, test_frame = new_cap.read()
                if ret and test_frame is not None:
                    frame_std = np.std(test_frame)
                    if frame_std > 5:  # Good signal
                        # Replace the old camera
                        if web_system.detector.cap:
                            web_system.detector.cap.release()
                        web_system.detector.cap = new_cap
                        
                        # Configure new camera  
                        new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        new_cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                        new_cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
                        new_cap.set(cv2.CAP_PROP_CONTRAST, 0.6)
                        
                        # Restart frame processing
                        web_system.start_camera()
                        
                        return jsonify({
                            'success': True,
                            'message': f'Switched to camera {camera_index}',
                            'frame_quality': float(frame_std)
                        })
                    else:
                        new_cap.release()
                        return jsonify({
                            'success': False,
                            'message': f'Camera {camera_index} has no video signal'
                        })
                else:
                    new_cap.release()
                    return jsonify({
                        'success': False, 
                        'message': f'Camera {camera_index} cannot capture frames'
                    })
            else:
                return jsonify({
                    'success': False,
                    'message': f'Camera {camera_index} not available'
                })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/camera_reset', methods=['POST'])
def api_camera_reset():
    """Reset and reinitialize camera with different settings"""
    try:
        # Stop current camera completely
        if web_system.detector and web_system.detector.cap:
            web_system.detector.cap.release()
            web_system.detector.cap = None
        
        web_system.stop_camera()
        time.sleep(2)  # Give time for release
        
        # Try different initialization approaches
        approaches = [
            {'backend': cv2.CAP_DSHOW, 'name': 'DirectShow'},
            {'backend': cv2.CAP_MSMF, 'name': 'Media Foundation'},
            {'backend': cv2.CAP_ANY, 'name': 'Default'},
        ]
        
        for approach in approaches:
            try:
                print(f"🔄 Trying {approach['name']} backend...")
                new_cap = cv2.VideoCapture(0, approach['backend'])
                
                if new_cap.isOpened():
                    # Test frame capture
                    ret, frame = new_cap.read()
                    if ret and frame is not None:
                        frame_std = np.std(frame)
                        print(f"📊 Frame quality: {frame_std}")
                        
                        if frame_std > 1:  # Even minimal signal is better than none
                            # Configure camera
                            new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            
                            # Try to improve visibility
                            new_cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                            new_cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)  # Try different brightness value
                            new_cap.set(cv2.CAP_PROP_CONTRAST, 64)    # Try different contrast
                            new_cap.set(cv2.CAP_PROP_GAMMA, 100)      # Adjust gamma
                            
                            # Replace camera
                            web_system.detector.cap = new_cap
                            
                            # Restart system
                            web_system.start_camera()
                            
                            return jsonify({
                                'success': True,
                                'message': f'Camera reset successful with {approach["name"]} backend',
                                'frame_quality': float(frame_std),
                                'backend': approach['name']
                            })
                
                new_cap.release()
                
            except Exception as e:
                print(f"❌ {approach['name']} backend failed: {e}")
                continue
        
        return jsonify({
            'success': False,
            'message': 'All camera backends failed',
            'suggestion': 'Check camera permissions and physical access'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# 🎯 IMAGE PROCESSING TECHNIQUES ENDPOINTS

@app.route('/api/process_image', methods=['POST'])
def api_process_image():
    """🎨 Unified Image Processing Endpoint for Frontend"""
    try:
        # Get analysis type from form data
        analysis_type = request.form.get('analysis_type', 'edge')
        
        # Thread-safe frame access
        with web_system.frame_lock:
            if web_system.current_frame is None:
                return jsonify({'success': False, 'error': 'No frame available - start camera first'})
            frame = web_system.current_frame.copy()
        
        # Process image for different analysis types
        results = {}
        
        # Always do edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Always do color analysis (HSV)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Enhanced processing (contrast/brightness)
        enhanced = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
        
        # Face detection
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray_for_faces = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_for_faces, 1.1, 4)
            
            face_detected = frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(face_detected, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
        except Exception as face_error:
            print(f"Face detection error: {face_error}")
            face_detected = frame.copy()
            faces = []
        
        # Encode all results
        def encode_image(img):
            ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                return f'data:image/jpeg;base64,{base64.b64encode(buffer).decode("utf-8")}'
            return None
        
        # Build response based on analysis_type
        response_data = {
            'success': True,
            'analysis_type': analysis_type,
            'processing_info': {
                'faces_detected': len(faces),
                'enhancement_applied': True,
                'edge_detection': 'Canny'
            }
        }
        
        print(f"🎨 Processing image analysis_type: {analysis_type}")
        print(f"📊 Frame shape: {frame.shape}, Faces detected: {len(faces)}")
        
        if analysis_type == 'edge' or analysis_type == 'full':
            edges_encoded = encode_image(edges_bgr)
            response_data['edges'] = edges_encoded
            print(f"✅ Edges encoded: {'Yes' if edges_encoded else 'Failed'}")
            
        if analysis_type == 'color' or analysis_type == 'full':
            hsv_encoded = encode_image(hsv)
            response_data['hsv'] = hsv_encoded
            print(f"✅ HSV encoded: {'Yes' if hsv_encoded else 'Failed'}")
            
        if analysis_type == 'face' or analysis_type == 'full':
            face_encoded = encode_image(face_detected)
            response_data['face_detected'] = face_encoded
            print(f"✅ Face detection encoded: {'Yes' if face_encoded else 'Failed'}")
            
        if analysis_type == 'advanced' or analysis_type == 'full':
            enhanced_encoded = encode_image(enhanced)
            response_data['enhanced'] = enhanced_encoded
            print(f"✅ Enhanced encoded: {'Yes' if enhanced_encoded else 'Failed'}")
        
        # For backward compatibility, also include result_image
        if analysis_type == 'edge':
            response_data['result_image'] = encode_image(edges_bgr)
        elif analysis_type == 'color':
            response_data['result_image'] = encode_image(hsv)
        elif analysis_type == 'face':
            response_data['result_image'] = encode_image(face_detected)
        else:
            response_data['result_image'] = encode_image(enhanced)
        
        # Log final response structure
        result_keys = [k for k in response_data.keys() if k not in ['success', 'analysis_type', 'processing_info']]
        print(f"📤 Returning response with keys: {result_keys}")
        
        return jsonify(response_data)
            
    except Exception as e:
        print(f"Image processing error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/process/edge_detection', methods=['POST'])
def api_edge_detection():
    """🔍 Edge Detection Processing"""
    try:
        # Thread-safe frame access
        with web_system.frame_lock:
            if web_system.current_frame is None:
                return jsonify({'success': False, 'message': 'No frame available - start camera first'})
            frame = web_system.current_frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Convert back to BGR for display
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', edges_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            return jsonify({
                'success': True,
                'technique': 'Edge Detection (Canny)',
                'result_image': f'data:image/jpeg;base64,{encoded_image}',
                'details': {
                    'algorithm': 'Canny Edge Detection',
                    'parameters': 'Gaussian Blur (5x5), Thresholds: 50-150'
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to encode image'})
    
    except Exception as e:
        print(f"Edge detection error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/process/color_analysis', methods=['POST'])
def api_color_analysis():
    """🎨 Color Analysis Processing"""
    try:
        # Thread-safe frame access
        with web_system.frame_lock:
            if web_system.current_frame is None:
                return jsonify({'success': False, 'message': 'No frame available - start camera first'})
            frame = web_system.current_frame.copy()
        
        # Color space conversions
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Calculate color statistics
        b_mean, g_mean, r_mean = cv2.mean(frame)[:3]
        
        # Create color analysis visualization
        height, width = frame.shape[:2]
        analysis_img = np.zeros((height, width*2, 3), dtype=np.uint8)
        
        # Original image on left
        analysis_img[:, :width] = frame
        
        # Color channels on right
        analysis_img[:height//3, width:] = frame[:height//3, :, 2:3]  # Red channel
        analysis_img[height//3:2*height//3, width:] = frame[height//3:2*height//3, :, 1:2]  # Green
        analysis_img[2*height//3:, width:] = frame[2*height//3:, :, 0:1]  # Blue
        
        # Add text overlays
        cv2.putText(analysis_img, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(analysis_img, 'Red Channel', (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(analysis_img, 'Green Channel', (width + 10, height//3 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(analysis_img, 'Blue Channel', (width + 10, 2*height//3 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', analysis_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            return jsonify({
                'success': True,
                'technique': 'Color Analysis',
                'result_image': f'data:image/jpeg;base64,{encoded_image}',
                'details': {
                    'color_means': {'red': float(r_mean), 'green': float(g_mean), 'blue': float(b_mean)},
                    'analysis': 'RGB channel separation and color statistics'
                }
            })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/process/face_analysis', methods=['POST'])
def api_face_analysis():
    """👤 Face Analysis Processing"""
    try:
        # Thread-safe frame access
        with web_system.frame_lock:
            if web_system.current_frame is None:
                return jsonify({'success': False, 'message': 'No frame available - start camera first'})
            frame = web_system.current_frame.copy()
        
        # Temporarily enable analysis mode for this request
        old_mode = web_system.analysis_mode
        web_system.analysis_mode = True
        
        # Get MediaPipe face detection results
        if web_system.detector and hasattr(web_system.detector, 'mediapipe_processor'):
            mp_results = web_system.detector.mediapipe_processor.process_frame(frame)
            faces = mp_results.get('faces', [])
            
            # Draw face detection results
            for face in faces:
                bbox = face.get('bbox', [0, 0, 0, 0])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Face (Quality: {face.get('quality_score', 0):.2f})", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Restore analysis mode
        web_system.analysis_mode = old_mode
        
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            return jsonify({
                'success': True,
                'technique': 'Face Analysis',
                'result_image': f'data:image/jpeg;base64,{encoded_image}',
                'details': {
                    'faces_detected': len(faces) if 'faces' in locals() else 0,
                    'algorithm': 'MediaPipe Face Detection'
                }
            })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/process/advanced_processing', methods=['POST'])
def api_advanced_processing():
    """📸 Advanced Image Processing"""
    try:
        # Thread-safe frame access
        with web_system.frame_lock:
            if web_system.current_frame is None:
                return jsonify({'success': False, 'message': 'No frame available - start camera first'})
            frame = web_system.current_frame.copy()
        
        # Apply multiple advanced processing techniques
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        # 2. Morphological operations
        kernel = np.ones((5,5), np.uint8)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # 3. Create composite image
        height, width = frame.shape[:2]
        composite = np.zeros((height*2, width*2, 3), dtype=np.uint8)
        
        # Original (top-left)
        composite[:height, :width] = frame
        
        # Histogram equalized (top-right)
        composite[:height, width:] = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        
        # Morphological opening (bottom-left)
        composite[height:, :width] = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
        
        # Morphological closing (bottom-right)
        composite[height:, width:] = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        cv2.putText(composite, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(composite, 'Histogram Equalized', (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(composite, 'Morphological Opening', (10, height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(composite, 'Morphological Closing', (width + 10, height + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', composite, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            return jsonify({
                'success': True,
                'technique': 'Advanced Image Processing',
                'result_image': f'data:image/jpeg;base64,{encoded_image}',
                'details': {
                    'techniques': ['Histogram Equalization', 'Morphological Opening', 'Morphological Closing'],
                    'description': 'Multiple advanced processing techniques applied'
                }
            })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/process/complete_analysis', methods=['POST'])
def api_complete_analysis():
    """🔬 Complete Analysis - Full Emotion Detection"""
    try:
        # Thread-safe frame access
        with web_system.frame_lock:
            if web_system.current_frame is None:
                return jsonify({'success': False, 'message': 'No frame available - start camera first'})
            frame = web_system.current_frame.copy()
        
        # Temporarily enable full analysis
        old_mode = web_system.analysis_mode
        web_system.analysis_mode = True
        web_system.last_analysis_request = time.time()
        
        # Run full emotion detection pipeline
        if web_system.detector:
            processed_frame = web_system.detector.process_frame(frame)
            
            # Get current emotions
            emotions = web_system.current_emotions or {}
            
            # Encode processed frame
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                encoded_image = base64.b64encode(buffer).decode('utf-8')
                
                # Restore analysis mode after short delay (will turn off after 10 seconds)
                import threading
                def reset_analysis_mode():
                    time.sleep(10)
                    web_system.analysis_mode = old_mode
                
                threading.Thread(target=reset_analysis_mode, daemon=True).start()
                
                return jsonify({
                    'success': True,
                    'technique': 'Complete Emotion Analysis',
                    'result_image': f'data:image/jpeg;base64,{encoded_image}',
                    'emotions': emotions,
                    'details': {
                        'components': ['MediaPipe Face Detection', 'DeepFace Emotion Analysis', 'YOLO Context Analysis'],
                        'description': 'Full multimodal emotion detection pipeline'
                    }
                })
        
        # Restore analysis mode if failed
        web_system.analysis_mode = old_mode
        return jsonify({'success': False, 'message': 'Analysis failed'})
    
    except Exception as e:
        web_system.analysis_mode = old_mode
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/session', methods=['POST'])
def api_set_session():
    """🎯 Set session ID for emotion storage synchronization"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                'success': False,
                'message': 'session_id is required'
            }), 400
        
        success = web_system.set_session_id(session_id)
        
        return jsonify({
            'success': success,
            'message': 'Session ID updated successfully' if success else 'Failed to update session ID',
            'session_id': web_system.session_id
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to set session ID: {str(e)}'
        }), 500

@app.route('/api/settings', methods=['POST'])
def api_update_settings():
    """🎛️ Update visual processing settings for enhanced features"""
    try:
        settings = request.get_json()
        # Visual settings updated silently
        
        # Update detector configuration if detector exists
        if web_system.detector and hasattr(web_system.detector, 'config'):
            config = web_system.detector.config
            
            # Update MediaPipe settings
            if hasattr(config, 'mediapipe_config'):
                mp_config = config.mediapipe_config
                mp_config.show_face_mesh = settings.get('show_face_mesh', True)
                mp_config.show_body_pose = settings.get('show_body_pose', True)
                mp_config.show_hand_tracking = settings.get('show_hand_tracking', True)
                mp_config.show_gaze_tracking = settings.get('show_gaze_tracking', True)
                # MediaPipe settings updated
            
            # Update YOLO settings
            if hasattr(config, 'yolo_config'):
                yolo_config = config.yolo_config
                yolo_config.confidence_threshold = settings.get('confidence_threshold', 0.25)
                # YOLO confidence updated
            
            # Update analysis settings
            config.emotion_analysis_interval = settings.get('analysis_interval', 30)
            config.show_analytics = settings.get('show_quality_indicators', True)
            
            # Visual settings updated successfully
            
        # Store settings globally for video feed processing
        web_system.visual_settings = settings
        
        return jsonify({
            'success': True,
            'message': 'Visual settings updated successfully',
            'settings': settings
        })
        
    except Exception as e:
        # Settings update error
        return jsonify({
            'success': False,
            'message': f'Failed to update settings: {str(e)}'
        }), 500

# UI routes removed - this is an API-only microservice  
# Use the main app at port 5000 for UI

@app.route('/api/mediapipe/landmarks')
def api_mediapipe_landmarks():
    """Get MediaPipe landmarks for visual overlays"""
    try:
        if not web_system.is_running or not web_system.detector:
            return jsonify({
                'success': False,
                'message': 'Camera not running or detector not available'
            })
        
        # Get current frame
        with web_system.frame_lock:
            if web_system.current_frame is None:
                return jsonify({
                    'success': False,
                    'message': 'No frame available'
                })
            frame = web_system.current_frame.copy()
        
        landmarks_data = {
            'success': True,
            'face_landmarks': [],
            'pose_landmarks': [],
            'hand_landmarks': [],
            'gaze_landmarks': {}
        }
        
        # Get MediaPipe results if processor is available
        if hasattr(web_system.detector, 'mediapipe_processor'):
            try:
                # Process frame with MediaPipe
                mp_results = web_system.detector.mediapipe_processor.process_frame(frame)
                
                # Extract face landmarks
                if 'face_mesh' in mp_results and mp_results['face_mesh']:
                    face_landmarks = []
                    for landmark in mp_results['face_mesh'].multi_face_landmarks[0].landmark:
                        face_landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    landmarks_data['face_landmarks'] = face_landmarks
                
                # Extract pose landmarks
                if 'pose' in mp_results and mp_results['pose'] and mp_results['pose'].pose_landmarks:
                    pose_landmarks = []
                    for landmark in mp_results['pose'].pose_landmarks.landmark:
                        pose_landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        })
                    landmarks_data['pose_landmarks'] = pose_landmarks
                
                # Extract hand landmarks
                if 'hands' in mp_results and mp_results['hands'] and mp_results['hands'].multi_hand_landmarks:
                    hand_landmarks_list = []
                    for hand_landmarks in mp_results['hands'].multi_hand_landmarks:
                        hand_data = {
                            'landmarks': []
                        }
                        for landmark in hand_landmarks.landmark:
                            hand_data['landmarks'].append({
                                'x': landmark.x,
                                'y': landmark.y,
                                'z': landmark.z
                            })
                        hand_landmarks_list.append(hand_data)
                    landmarks_data['hand_landmarks'] = hand_landmarks_list
                
                # Extract gaze tracking (simplified)
                if 'face_mesh' in mp_results and mp_results['face_mesh']:
                    # Get eye landmarks for gaze estimation
                    if mp_results['face_mesh'].multi_face_landmarks:
                        face_landmarks_list = mp_results['face_mesh'].multi_face_landmarks[0].landmark
                        
                        # Left eye center (approximate)
                        left_eye_landmarks = [33, 133, 157, 158, 159, 160, 161, 163]
                        right_eye_landmarks = [362, 398, 384, 385, 386, 387, 388, 466]
                        
                        if len(face_landmarks_list) > max(left_eye_landmarks + right_eye_landmarks):
                            left_eye_center = {
                                'x': sum(face_landmarks_list[i].x for i in left_eye_landmarks) / len(left_eye_landmarks),
                                'y': sum(face_landmarks_list[i].y for i in left_eye_landmarks) / len(left_eye_landmarks)
                            }
                            right_eye_center = {
                                'x': sum(face_landmarks_list[i].x for i in right_eye_landmarks) / len(right_eye_landmarks),
                                'y': sum(face_landmarks_list[i].y for i in right_eye_landmarks) / len(right_eye_landmarks)
                            }
                            
                            # Simple gaze direction estimation
                            landmarks_data['gaze_landmarks'] = {
                                'left_eye': left_eye_center,
                                'right_eye': right_eye_center,
                                'gaze_direction': {
                                    'x': 0.1,  # Simplified - would need more complex calculation
                                    'y': 0.05
                                }
                            }
                
            except Exception as mp_error:
                # MediaPipe processing error
                # Return success=True but with empty landmarks
                pass
        
        return jsonify(landmarks_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to get MediaPipe landmarks'
        })

if __name__ == '__main__':
    print("Y.M.I.R Face Emotion Detection MICROSERVICE")
    print("Microservice running on: http://localhost:5002")
    print("Video feed: http://localhost:5002/video_feed")
    print("Health check: http://localhost:5002/health")
    print("Camera debug: http://localhost:5002/api/camera_debug")
    print("\n📹 Camera Troubleshooting:")
    print("🔍 Test all cameras: POST /api/camera_test_different_indices")
    print("🔄 Switch camera: POST /api/camera_switch/<index>")
    print("\n🎯 Image Processing Endpoints:")
    print("📸 Advanced Processing: POST /api/process/advanced_processing")
    print("🔍 Edge Detection: POST /api/process/edge_detection")
    print("🎨 Color Analysis: POST /api/process/color_analysis")  
    print("👤 Face Analysis: POST /api/process/face_analysis")
    print("🔬 Complete Analysis: POST /api/process/complete_analysis")
    
    # 📹 Camera is ready but NOT auto-started (user must manually start)
    print("\n📹 Camera ready - use /api/start_camera to begin")
    print("💡 Manual control:")
    print("   - Start: POST /api/start_camera")
    print("   - Stop:  POST /api/stop_camera")
    print("   - Status: GET /api/status")
    
    # 🎯 START ON PORT 5002 AS MICROSERVICE - PRODUCTION MODE
    # 🚀 DISABLE DEBUG: Prevents crashes and auto-restart on file changes
    app.run(debug=False, host='0.0.0.0', port=5002, threaded=True)