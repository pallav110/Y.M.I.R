# Y.M.I.R: Yielding Melodies for Internal Restoration 🎵🧠✨

_A Comprehensive AI-Powered Mental Health & Wellness Platform with Multimodal Emotion Detection, Music Therapy, Brainwave Entrainment & Microservices Architecture_

[![Python Version](https://img.shields.io/badge/python-3.9.5-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/flask-3.1.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Microservices](https://img.shields.io/badge/architecture-microservices-orange)]()
[![Gemini AI](https://img.shields.io/badge/ai-gemini-yellow)]()

![Y.M.I.R Banner](https://github.com/pallav110/DTI-AND-AI-PROJECT/blob/main/YMIR%20thumbnail.jpg?raw=true)

---

## 🎥 **Demo & Documentation**

<div align="center">
  <a href="https://youtu.be/qzxqyhuB_XU?si=i65ziCP-z99yOAIZ" target="_blank">
    <img src="https://img.shields.io/badge/Watch%20Demo%20Video-YouTube-red?logo=youtube&style=for-the-badge" alt="Watch on YouTube" height="35">
  </a>
  <br>
  <em>📄 Complete presentation available in the repository: <strong>YMIR_Final_PPT.pptx</strong></em>
</div>

---

## 🌟 **Project Overview**

**Y.M.I.R** (Yielding Melodies for Internal Restoration) is a cutting-edge, AI-powered mental health and wellness platform that combines multiple therapeutic modalities to provide personalized emotional support. Our system intelligently detects emotions through multiple channels — facial expressions, text sentiment, and conversational context — and delivers targeted interventions through music therapy, brainwave entrainment (binaural beats & isochronic tones), guided exercises, gamified experiences, and community support.

The platform has been refactored from a 2,071-line monolithic Flask application into a production-grade **microservices architecture** with 7 independent services, ensemble machine learning models, Gemini AI integration with automatic API key rotation, Firebase authentication, and an experimental brainwave entrainment lab.

### 🎯 **Mission Statement**

To revolutionize mental health care by making personalized therapeutic interventions accessible, engaging, and effective through the power of artificial intelligence, music therapy, and neuroscience-based audio entrainment.

---

## 🚀 **Core Features**

### 🧠 **AI-Powered Multi-Modal Emotion Detection**

- **Facial Expression Recognition**: Enhanced v3 system using MediaPipe, YOLO, and DeepFace ensemble for robust emotion analysis
- **Face Quality Assessment**: Multi-factor scoring (size, sharpness, brightness) — POOR/FAIR/GOOD/EXCELLENT classification
- **Multi-Face Detection**: Up to 3 faces simultaneously with per-face emotion tracking
- **Emotion Smoothing & Stability**: Temporal analysis prevents rapid emotion fluctuations; 5-frame moving average filter
- **Environmental Context Analysis**: YOLO-based scene understanding (YOLOv11/v10/v9/v8 variants) amplifies emotion detection accuracy based on user context
- **True ML Emotion Context**: Unsupervised learning system using KMeans clustering (8 environment + 6 emotion clusters), MLPRegressor neural network (100-50-25), PCA dimensionality reduction, and TF-IDF vectorization — discovers emotion-environment patterns purely from data without hardcoded rules
- **Separate Dedicated Web Apps**: Standalone `facewebapp.py` and `chatbotwebapp.py` each run independent Flask servers with their own video feeds, API endpoints, emotion analytics, and storage capabilities — deployable independently for distributed processing
- **Advanced NLP**: Transformer models (DistilBERT, RoBERTa) for text-based emotion analysis with ensemble voting
- **Context-Aware Chatbot**: Streaming responses like ChatGPT with conversation history, encryption, and function calling
- **Emotion Fusion Algorithm**: 5 fusion strategies (simple, adaptive, confidence-based, temporal weighted, weighted average) to integrate visual and textual emotional signals

### 🤖 **Ensemble Machine Learning Pipeline**

- **Multiple Base Models**: RandomForest, GradientBoosting, SVC, MLPClassifier, XGBoost, LightGBM
- **Ensemble Architectures**: VotingClassifier (soft/hard), StackingClassifier for meta-learning
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Cross-Validation**: StratifiedKFold with 5 folds for robust evaluation
- **Feature Engineering**: 14 audio features (Tempo, Energy, Valence, Danceability, Acousticness, Loudness, Mode, Speechiness, Instrumentalness, Liveness, Key, Popularity, Duration)
- **Fixed Overfitting**: Removed data leakage; achieved realistic 65-75% accuracy target
- **10 Emotion Categories**: Joy, Sadness, Excitement, Calm, Anger, Fear, Surprise, Nostalgia, Love, Neutral

### 🌐 **Gemini AI Integration**

- **API Key Rotation Manager**: Auto-cycles through 8+ Gemini API keys when quotas are exhausted
- **Automatic Failover**: Seamless rotation to next available key with error tracking
- **Quota Recovery**: 24-hour reset window with persistent state management
- **Therapeutic Recommendations**: AI-powered personalized wellness suggestions
- **Contextual Chat Responses**: Gemini-enhanced conversational therapy with streaming output

### 🎵 **Intelligent Music Therapy**

- **Personalized Recommendations**: ML-powered music matching based on emotional states with 4 strategies:
  - *Therapeutic*: Supports emotional healing and mental health benefits
  - *Mood Matching*: Reflects current emotional state
  - *Mood Regulation*: Guides toward positive emotional state
  - *Adaptive*: Intelligently combines strategies based on context
- **Dynamic Playlist Generation**: Real-time adaptation to mood changes via content-based filtering
- **Therapeutic Music Database**: 3,200+ curated songs with detailed audio features, mental health benefits, and musical feature analysis
- **Multi-Platform Integration**: YouTube, SoundCloud integration for seamless streaming

### 🧘 **Comprehensive Wellness Tools**

- **Guided Meditation**: Personalized meditation sessions based on current mood (anxious, stressed, tired, happy)
- **Breathing Exercises**: Adaptive breathing techniques based on stress levels with visual guidance
- **AI-Powered Journaling**: Sentiment analysis with mood trend tracking and pattern insights
- **Goal Setting & Tracking**: Personal wellness goal management with streak tracking and habit monitoring
- **Sound Therapy**: Ambient soundscapes for relaxation, focus, and sleep
- **Mood-Based Recommendations**: Movies, wellness activities, and self-care suggestions

### 🧠 **Brainwave Entrainment Lab** (Experimental)

- **Binaural Beats**: Two different frequencies presented to each ear — the brain perceives a third "beat" frequency. Requires stereo headphones.
- **Isochronic Tones**: Single amplitude-modulated tone — works with any audio setup (speakers or headphones)
- **6 Brainwave Bands**:
  | Band | Frequency | Target State |
  |------|-----------|-------------|
  | Delta | 0.5–4 Hz | Deep sleep, physical healing |
  | Theta | 4–8 Hz | Meditation, creativity, deep relaxation |
  | Alpha | 8–12 Hz | Calm focus, anxiety relief, stress reduction |
  | Beta | 13–30 Hz | Concentration, mental clarity, active thinking |
  | Gamma | 30–50 Hz | Peak performance, cognitive enhancement |
  | Epsilon | 100 Hz | Advanced meditative states |
- **6 Wellness Goals**: Deep Sleep, Enhanced Focus, Anxiety Relief, Deep Relaxation, Stress Reduction, Creative Flow
- **Generated Audio**: 36 WAV files (18 binaural + 18 isochronic, 3 variations × 6 bands)
- **Spotify Dataset**: 10,000+ binaural beats tracks scraped and categorized by brainwave type and purpose
- **Synthetic Generation**: Python-based audio synthesis with random variations for each session

### 🎮 **AI Gaming Hub**

- **Emotion-Based Gaming**: Game recommendations driven by real-time emotional state detection
- **AI NPC "NOVA"**: Fully frontend-implemented interactive game recommendation chatbot with personality-driven dialogue, emotion-aware game suggestions, and dynamic response system (no backend dependency)
- **4 Story Themes**: Fantasy, Sci-Fi, Cyberpunk, Medieval for AI-generated narratives
- **4 Emotion-Based Styles**: Excited, Calm, Focused, Competitive
- **3D Visual Environment**: Three.js particle system with game-HUD aesthetic
- **Profile System**: Tracks game style preference, difficulty, and play history

### 🤝 **Community & Support**

- **Community Forum**: Safe space for sharing experiences and support
- **Professional Therapist Finder**: Advanced search with real-time location detection and international database integration (NPI Registry US, Practo India, Government Health Portals)
- **Global Mental Health Directory**: Access to verified therapists worldwide with automatic geolocation
- **Crisis Support**: 24/7 helpline integration and emergency resources
- **Peer Support Network**: Connect with others on similar wellness journeys
- **Anonymous Posting**: Privacy-preserving community participation

### 📊 **Analytics & Insights**

- **Emotion Timeline**: Visual representation of emotional patterns over time
- **Mood Transition Analysis**: Understanding emotional triggers and patterns
- **Progress Tracking**: Detailed wellness metrics and improvement indicators
- **Personal Dashboard**: Comprehensive overview of mental health journey with session analytics
- **Real-Time Emotion Monitoring**: Live webcam-based tracking with confidence metrics

### 🔐 **Security & Authentication**

- **Firebase Authentication**: Secure user management with email/password and Google OAuth
- **Session-Based Isolation**: GDPR-compliant data storage per user session
- **Encrypted Chat Storage**: Cryptography-based message encryption for sensitive conversations
- **User Consent Management**: Privacy mode toggle, local-only processing options
- **Input Sanitization**: XSS prevention and data validation across all endpoints

---

## 🏗️ **Technical Architecture**

### **System Architecture**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Y.M.I.R Platform                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐ │
│  │   Frontend    │   │   Backend    │   │     AI Services          │ │
│  │  (HTML/CSS/JS)│◄─►│  (Flask)     │◄─►│  ┌───────────────────┐  │ │
│  │               │   │  Orchestrator│   │  │Ensemble ML Models  │  │ │
│  └──────────────┘   └──────┬───────┘   │  │(RF/GBM/SVC/MLP/    │  │ │
│                            │            │  │ XGBoost/LightGBM)  │  │ │
│                            │            │  └───────────────────┘  │ │
│                            ▼            │  ┌───────────────────┐  │ │
│                    ┌───────────────┐    │  │Transformers NLP   │  │ │
│                    │  Microservices │    │  │(DistilBERT/       │  │ │
│                    │  ┌─────────┐   │    │  │ RoBERTa Ensemble) │  │ │
│                    │  │Emotion  │   │    │  └───────────────────┘  │ │
│                    │  │Detection│   │    │  ┌───────────────────┐  │ │
│                    │  ├─────────┤   │    │  │DeepFace Ensemble  │  │ │
│                    │  │Computer │   │    │  │(VGG-Face/FaceNet/ │  │ │
│                    │  │Vision   │   │    │  │  Default)         │  │ │
│                    │  ├─────────┤   │    │  └───────────────────┘  │ │
│                    │  │Music &  │   │    │  ┌───────────────────┐  │ │
│                    │  │Audio    │   │    │  │  Gemini AI        │  │ │
│                    │  ├─────────┤   │    │  │(8 Key Rotation)   │  │ │
│                    │  │Recommen-│   │    │  └───────────────────┘  │ │
│                    │  │dation   │   │    │  ┌───────────────────┐  │ │
│                    │  ├─────────┤   │    │  │Binaural/          │  │ │
│                    │  │Wellness │   │    │  │Isochronic Synth   │  │ │
│                    │  ├─────────┤   │    │  └───────────────────┘  │ │
│                    │  │Text     │   │    └──────────────────────────┘ │
│                    │  │Analysis │   │                                  │
│                    │  ├─────────┤   │                                  │
│                    │  │Web      │   │                                  │
│                    │  │Routes   │   │                                  │
│                    │  └─────────┘   │                                  │
│                    └───────┬───────┘                                  │
│                            │                                          │
│                            ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                    External Integrations                      │    │
│  │  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌───────────────┐  │    │
│  │  │ Firebase │  │ YouTube  │  │Spotify │  │ Therapist     │  │    │
│  │  │ Auth/DB  │  │ Data API │  │  API   │  │ Databases     │  │    │
│  │  └──────────┘  └──────────┘  └────────┘  └───────────────┘  │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### **Microservices Architecture**

The original 2,071-line monolithic `app.py` was refactored into extracted service modules (referenced from `DOCS/MICROSERVICES_SUMMARY.md`) plus 3 independently deployable microservice servers, all orchestrated by the system launcher:

| Service/Module | File | Port | Lines | Responsibility |
|----------------|------|------|-------|----------------|
| **Face Microservice** | `face_microservice_working.py` | 5002 | ~1,789 | Real-time facial emotion detection with MediaPipe, YOLO, DeepFace ensemble; Firebase storage; emotion smoothing; video streaming |
| **Text Microservice** | `text_microservice_working.py` | 5003 | ~2,367 | Multi-model NLP (DistilBERT, RoBERTa ensemble), Gemini chat integration, streaming responses, conversation history, encryption |
| **Chatbot Web App** | `chatbotwebapp.py` | 5001 | ~1,277 | Standalone chatbot server with Gemini API, production emotion analyzer, streaming chat interface, fallback chatbot |
| **Face Web App** | `facewebapp.py` | 5000 | ~709 | Independent face detection web app with Firebase storage, emotion analytics, session export |
| **Extracted Emotion Detection** | *(service module)* | — | ~200 | AI/ML emotion processing, sentiment analysis, negation handling |
| **Extracted Computer Vision** | *(service module)* | — | ~120 | Video processing, face mesh, gaze, body/hand landmarks |
| **Extracted Music & Audio** | *(service module)* | — | ~930 | Music search/download, audio streaming |
| **Extracted Recommendation** | *(service module)* | — | ~180 | ML-based song recommendations |
| **Extracted Wellness** | *(service module)* | — | ~320 | Journal analysis, breathing, goals, meditation |
| **Extracted Text Analysis** | *(service module)* | — | ~80 | Chatbot response generation |
| **Extracted Web Routes** | *(service module)* | — | ~350 | Render functions, favorites, dashboard |
| **Orchestrator** | `start_ymir.py` | — | 207 | Launches all microservice servers in sequence with health checks |

**Key Metrics**:
- 85% size reduction in main application (2,071 → 300 lines)
- 600% improvement in team scalability (1 → 6+ developers)
- 90% risk reduction in deployments (service-by-service)
- 75% faster debugging time per service

### **Technology Stack**

#### **Backend Technologies**
- **Framework**: Flask 3.1.0 (Python web framework)
- **Database**: SQLAlchemy with SQLite (development), Firebase Firestore (production)
- **Session Management**: Flask-Session, Firebase Auth sessions
- **Email Integration**: Flask-Mail for notifications and communications
- **CORS**: Flask-CORS for cross-origin microservice communication

#### **AI & Machine Learning**
- **Computer Vision**: OpenCV 4.11.0, MediaPipe 0.10.21, YOLO for object/context detection
- **Facial Recognition**: DeepFace 0.0.93 (VGG-Face, FaceNet, Default ensemble), dlib 19.24.6
- **NLP Models**: Transformers 4.48.3 (HuggingFace) — DistilBERT, RoBERTa
- **Sentiment Analysis**: VADER Sentiment 3.3.2
- **Ensemble ML Framework**: scikit-learn 1.6.1, XGBoost, LightGBM, imbalanced-learn (SMOTE)
- **Deep Learning**: PyTorch 2.6.0
- **LLM Integration**: Google Generative AI (Gemini) with automatic key rotation

#### **Audio Processing**
- **Binaural/Isochronic Generation**: NumPy, SciPy, soundfile, simpleaudio
- **Audio Download**: yt-dlp, Spotify API (Spotipy)
- **Audio Features**: Spotify audio feature analysis (danceability, energy, valence, tempo, etc.)

#### **Data Processing**
- **Data Manipulation**: Pandas 2.2.3, NumPy 1.26.3
- **Scientific Computing**: SciPy 1.13.1

#### **Frontend Technologies**
- **Styling**: Tailwind CSS, Custom CSS with glassmorphism + cyberpunk design system
- **JavaScript**: Vanilla JS with modern ES6+ features, WebSocket-ready
- **UI Components**: Custom glassmorphism design system with holographic borders
- **Animations**: GSAP, Three.js (gaming hub 3D), CSS animations, particle effects
- **Fonts**: Orbitron (gaming), Poppins (modern), Inter (tech)

#### **Cloud & Infrastructure**
- **Authentication**: Firebase Admin SDK (Auth, Firestore)
- **Deployment**: Vercel (serverless Python), Docker-ready microservices
- **File Storage**: Local with Firebase Firestore cloud storage
- **State Management**: JSON state files for scraper, preprocessor, Gemini key rotation

#### **External Integrations**
- **Music Services**: YouTube Data API, SoundCloud API, Spotify Web API
- **Healthcare Databases**: NPI Registry API (US), Practo API (India), Government Health Portals
- **Geolocation Services**: OpenStreetMap Nominatim API
- **International Therapy Platforms**: BetterHelp, Mental health directories worldwide
- **Communication**: SMTP email services

### **Machine Learning Pipeline**

#### **Ensemble Emotion Detection Flow**

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        MULTIMODAL INPUT LAYER                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────┐         ┌──────────────────┐                        │
│  │  Camera Input    │         │   Text Input      │                       │
│  │  (Webcam/Video)  │         │  (Chat/Messages)  │                       │
│  └────────┬────────┘         └────────┬─────────┘                        │
│           │                           │                                   │
│           ▼                           ▼                                   │
│  ┌─────────────────┐         ┌──────────────────┐                        │
│  │  MediaPipe Face  │         │  Tokenization     │                       │
│  │  Detection       │         │  + Preprocessing  │                       │
│  └────────┬────────┘         └────────┬─────────┘                        │
│           │                           │                                   │
│           ▼                           ▼                                   │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                    DETECTION LAYER                                │    │
│  │  ┌──────────────┐  ┌───────────────┐  ┌──────────────────────┐  │    │
│  │  │ DeepFace      │  │ YOLO Context  │  │ Transformer Ensemble │  │    │
│  │  │ Ensemble      │  │ Analyzer      │  │ (DistilBERT +        │  │    │
│  │  │(VGG-Face/     │  │ (Environment  │  │  RoBERTa)            │  │    │
│  │  │ FaceNet/      │  │  Detection)   │  │                      │  │    │
│  │  │ Default)      │  │               │  │                      │  │    │
│  │  └──────┬───────┘  └──────┬────────┘  └──────────┬───────────┘  │    │
│  └─────────┼─────────────────┼──────────────────────┼──────────────┘    │
│            │                 │                      │                   │
│            ▼                 ▼                      ▼                   │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                    FUSION LAYER (AdvancedEmotionFusionEngine)     │    │
│  │                                                                   │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐  │    │
│  │  │ Simple        │  │ Confidence-  │  │ Temporal Weighted      │  │    │
│  │  │ Fusion        │  │ Based Fusion │  │ Fusion                 │  │    │
│  │  └──────────────┘  └──────────────┘  └────────────────────────┘  │    │
│  │  ┌──────────────┐  ┌──────────────┐                              │    │
│  │  │ Weighted      │  │ Adaptive     │ ← DEFAULT                   │    │
│  │  │ Average       │  │ Fusion       │                             │    │
│  │  └──────────────┘  └──────────────┘                              │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                    ENSEMBLE ML LAYER                              │    │
│  │                                                                   │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐  │    │
│  │  │ RandomForest  │  │ Gradient     │  │ SVC (Support Vector)   │  │    │
│  │  │ Classifier    │  │ Boosting     │  │                        │  │    │
│  │  └──────────────┘  └──────────────┘  └────────────────────────┘  │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐  │    │
│  │  │ MLPClassifier │  │ XGBoost      │  │ LightGBM               │  │    │
│  │  │ (Neural Net)  │  │              │  │                        │  │    │
│  │  └──────────────┘  └──────────────┘  └────────────────────────┘  │    │
│  │                                                                   │    │
│  │  ┌────────────────────────────────────────────────────────────┐  │    │
│  │  │         VotingClassifier / StackingClassifier             │  │    │
│  │  │           (Soft Voting Ensemble)                          │  │    │
│  │  └────────────────────────────────────────────────────────────┘  │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                    RECOMMENDATION LAYER                           │    │
│  │                                                                   │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐  │    │
│  │  │ Therapeutic   │  │ Mood         │  │ Mood Regulation        │  │    │
│  │  │ Strategy      │  │ Matching     │  │ Strategy               │  │    │
│  │  └──────────────┘  └──────────────┘  └────────────────────────┘  │    │
│  │  ┌────────────────────────────────────────────────────────────┐  │    │
│  │  │              Adaptive Strategy (Default)                   │  │    │
│  │  │   Combines all strategies based on context & confidence     │  │    │
│  │  └────────────────────────────────────────────────────────────┘  │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                    OUTPUT LAYER                                   │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐  │    │
│  │  │ Music        │  │ Wellness     │  │ Binaural/Isochronic    │  │    │
│  │  │ Playlist     │  │ Recommendations│ │ Tone Recommendations   │  │    │
│  │  └──────────────┘  └──────────────┘  └────────────────────────┘  │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

#### **Music Recommendation Engine**

1. **Feature Extraction**: 14 audio features (tempo, energy, valence, danceability, acousticness, loudness, mode, speechiness, instrumentalness, liveness, key, popularity, duration)
2. **Emotion Mapping**: 10 emotion categories mapped to therapeutic music categories with ML confidence scoring
3. **Content-Based Filtering**: Match user emotions to song characteristics using trained ensemble models
4. **SMOTE Handling**: Synthetic oversampling for underrepresented emotion classes
5. **Stratified Cross-Validation**: 5-fold evaluation to prevent overfitting (fixed from previous 97% → realistic 65-75%)
6. **Dynamic Adjustment**: Real-time recommendation updates based on user feedback and emotion changes

#### **Brainwave Audio Generation Pipeline**

```
┌─────────────────────────────────────────────────────────────────────┐
│                  BRAINWAVE AUDIO GENERATION                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────┐    ┌──────────────────────┐                     │
│  │  Brainwave Band  │    │  Generation Method    │                    │
│  │  Selection       │    │  Selection            │                    │
│  │  (Delta/Theta/   │───►│  (Binaural /          │                    │
│  │   Alpha/Beta/    │    │   Isochronic)         │                    │
│  │   Gamma/Epsilon) │    └──────────┬───────────┘                     │
│  └─────────────────┘               │                                 │
│                                     ▼                                 │
│          ┌────────────────────────────────────────────┐               │
│          │           Audio Parameters                  │              │
│          │  ┌────────────┐    ┌────────────────────┐  │              │
│          │  │ Base Freq  │    │ Beat Frequency     │  │              │
│          │  │ (200-250Hz)│    │ (Band-specific)    │  │              │
│          │  └────────────┘    └────────────────────┘  │              │
│          │  ┌────────────┐    ┌────────────────────┐  │              │
│          │  │ Duration   │    │ Random Variations  │  │              │
│          │  │ (10-1000s) │    │ (±0.5Hz beat,      │  │              │
│          │  └────────────┘    │  ±10Hz carrier)    │  │              │
│          │                    └────────────────────┘  │              │
│          └────────────────────────────────────────────┘              │
│                               │                                       │
│                               ▼                                       │
│          ┌────────────────────────────────────────────┐               │
│          │          Waveform Synthesis                 │              │
│          │                                             │              │
│          │  Binaural:  left = sin(2π·f_left·t)        │              │
│          │             right = sin(2π·f_right·t)       │              │
│          │             where f_right = f_left + beat   │              │
│          │                                             │              │
│          │  Isochronic: signal = sin(2π·f_carrier·t)   │              │
│          │              × modulation(beat_freq)        │              │
│          └────────────────────────────────────────────┘              │
│                               │                                       │
│                               ▼                                       │
│          ┌────────────────────────────────────────────┐               │
│          │        Post-Processing                      │              │
│          │  • Amplitude modulation (0.5Hz gentle       │              │
│          │    tremolo)                                 │              │
│          │  • Fade-in/out envelope (20ms ramp)         │              │
│          │  • Stereo normalization                     │              │
│          │  • Random variation for unique sessions     │              │
│          └────────────────────────────────────────────┘              │
│                               │                                       │
│                               ▼                                       │
│          ┌────────────────────────────────────────────┐               │
│          │             Output                          │              │
│          │  • WAV file (44.1kHz, 16-bit stereo)        │              │
│          │  • Immediate audio playback                  │              │
│          │  • Served via Flask route (/experimental/)   │              │
│          └────────────────────────────────────────────┘              │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 **Project Structure**

```
Y.M.I.R/
├── 📁 api/                          # Vercel deployment entry point
│   └── index.py                    # Flask app wrapper for serverless
│
├── 📁 datasets/                     # ML datasets and training data
│   ├── therapeutic_music_enriched.csv   # 3,200+ enriched therapeutic songs
│   ├── Y.M.I.R. original dataset.csv    # 3,465 original research tracks
│   ├── binaural_beats_experimental.csv  # 10,000+ binaural/isochronic tracks
│   └── 📁 imagesofdataset/              # Visual analysis examples
│
├── 📁 data/                         # Application data stores
│   ├── movies.csv                  # Movie recommendations
│   └── posts.json                 # Community posts
│
├── 📁 templates/                    # HTML templates (29 pages)
│   ├── ai_dashboard.html           # Main AI emotion dashboard (10,681 lines)
│   ├── index.html                  # Main emotion detection interface
│   ├── home.html                   # Landing page
│   ├── dashboard.html              # User dashboard
│   ├── experimental_home.html      # Brainwave entrainment hub
│   ├── binaural_beats.html         # Binaural beats session page
│   ├── isochronic_beats.html       # Isochronic tones session page
│   ├── gaming.html                 # AI Gaming Hub (918 lines)
│   ├── meditation.html             # Meditation module
│   ├── meditation_result.html      # Meditation results
│   ├── journal.html                # Journaling interface
│   ├── breathing.html              # Breathing exercises
│   ├── goals.html                  # Goal setting
│   ├── sound_therapy.html          # Sound therapy interface
│   ├── wellness_tools.html         # Comprehensive wellness toolkit
│   ├── therapist_finder.html       # Professional therapist directory
│   ├── image_processing.html       # Advanced image analysis
│   ├── community_support.html      # Community features
│   ├── emotion_timeline.html       # Analytics dashboard
│   ├── recommendations.html        # Music recommendations
│   ├── about.html, contact.html, features.html, services.html,
│   ├── pricing.html, privacy.html, cookiepolicy.html
│   ├── header.html, footer.html
│   └── test.html
│
├── 📁 static/                      # Static assets
│   ├── styles.css                  # Main stylesheet
│   └── 📁 audio/                   # Audio files for therapy
│
├── 📁 enhancements/                # Enhanced production modules
│   └── 📁 src-new/
│       ├── 📁 face/               # Facial emotion detection
│       │   ├── fer_enhanced_v3.py # Advanced v3 emotion detector
│       │   ├── web_app.py         # Web-based emotion detection Flask app
│       │   ├── true_ml_emotion_context.py  # True ML context analyzer
│       │   ├── web_emotion_detection.html   # Standalone HTML for face detection (1053 lines, also at root)
│       │   ├── true_ml_emotion_context.py  # Unsupervised ML emotion-context learner (KMeans, MLPRegressor, PCA, TF-IDF) — 419 lines
│       │   └── 📁 face_models/   # Modular components
│       │       ├── __init__.py             # Exports MediaPipeProcessor, EnhancedYOLOProcessor, DeepFaceEnsemble, etc.
│       │       ├── mediapipemodel.py       # MediaPipe face mesh, pose, hand tracking, FaceQuality enum (541 lines)
│       │       ├── yolomodel.py            # YOLOv11/v10/v9/v8 processor + EmotionContextAnalyzer + Gemini classification (819 lines)
│       │       ├── deepfacemodel.py        # Thread-safe DeepFace ensemble (VGG-Face/FaceNet/Default backends) (505 lines)
│       │       └── 📁 true_ml_models/      # Trained pickle models from unsupervised learning
│       │
│       ├── 📁 textual/            # Text-based emotion detection
│       │   └── chatbot_production_ready.py  # Production chatbot
│       │
│       ├── 📁 multimodal_fusion/  # Emotion fusion engine
│       │   ├── unified_emotion_music_system.py  # Main fusion system (1,911 lines)
│       │   └── real_emotion_combiner.py         # Real-time combiner
│       │
│       ├── 📁 preprocess/         # Data preprocessing
│       │   ├── enhanced_scraper_production.py       # Spotify scraper
│       │   ├── enhanced_preprocess_production.py    # Emotion mapping
│       │   ├── binaural_beats_scraper.py            # Binaural beat scraper
│       │   ├── binaural_audio_downloader.py         # Audio downloader
│       │   └── ml_emotion_detector.py               # ML ensemble detector
│       │
│       └── 📁 recommendation/     # Music recommendation
│           ├── music_recommendation_production.py   # Production ML model
│           └── test_recommendations.py
│
├── 📁 experimental/               # Experimental audio generation
│   ├── binaural_audio_testing.py  # Binaural beat synthesizer
│   ├── isochronic_audio_testing.py # Isochronic tone synthesizer
│   ├── 📁 binaural_tone/          # 18 generated WAV files (3 variations × 6 bands)
│   ├── 📁 isochronic_tone/        # 18 generated WAV files (3 variations × 6 bands)
│   └── 📁 foreground_music/       # Optional background music tracks
│
├── 📁 DOCS/                       # Comprehensive documentation
│   ├── BUSINESS_RULES.md          # System business rules
│   ├── ENHANCED_IMPROVEMENTS.md   # v2.0 improvements
│   ├── IMPROVEMENTS_NEEDED.md     # Enterprise transformation plan
│   ├── IMPROVEMENTS_COMPARISON.md # Before vs after comparison
│   ├── MULTIMODAL_SYSTEM_INTEGRATION.md  # Multimodal system guide
│   ├── MICROSERVICES_SUMMARY.md   # Microservices extraction summary
│   ├── MICROSERVICES_INTEGRATION_PLAN.md  # Integration roadmap
│   ├── README_ENHANCED_SYSTEM.md  # Enhanced system guide
│   ├── FIREBASE_STORAGE_README.md # Firebase storage architecture
│   ├── THEME_MANAGE.md            # Visual design system (789 lines)
│   └── WEB_APP_COMPREHENSIVE_ARCHITECTURE.md  # Complete architecture
│
├── 📁 OFFICIAL_DOCUMENTS/         # Official project documents
│   ├── YMIR_Final_PPT.pptx        # Final presentation
│   ├── YMIR_Final_Report.pdf      # Final report
│   ├── SMART INDIA HACKATHON 2025_ Y.M.I.R..pdf
│   ├── IVP_PROJ_REPORT_DRAFT1.pdf
│   └── *.docx                     # Business docs
│
├── 📁 models/                     # Trained ML models (not in repo)
│   ├── ensemble_model.pkl         # Main emotion classifier
│   ├── label_encoder.pkl          # Label encoding
│   └── scaler.pkl                 # Feature scaling
│
├── 📁 TEAM_DOCUMENTATION/         # Team docs
├── 📁 TECHNICAL_DOCUMENTATION/    # Technical docs
│
├── app.py                         # Main Flask application (orchestrator)
├── app_original_backup.py         # Original 2,071-line monolith backup
│
├── face_microservice_working.py   # Face emotion microservice (port 5002)
├── text_microservice_working.py   # Text emotion microservice (port 5003)
├── facewebapp.py                  # Face web app
├── chatbotwebapp.py               # Chatbot web app (1,277 lines)
│
├── gemini_api_manager.py          # Gemini API key rotation manager (337 lines)
├── firebase_auth_backend.py       # Firebase auth backend
├── image_processing.py            # Image processing module (edge detection, enhancement)
├── web_emotion_detection.html     # Standalone face detection HTML (1053 lines, also in enhancements)
├── text_emotion_detection.html    # Standalone text emotion HTML
├── test_facial_firebase.py        # Integration tests for Firebase
├── test_rotation.py               # Gemini API key rotation test script
│
├── start_ymir.py                  # System launcher orchestrator
├── start.bat                      # Windows startup script
├── start.ps1                      # PowerShell startup script
│
├── requirements.txt               # Python dependencies
├── runtime.txt                    # Python version for deployment
├── vercel.json                    # Vercel deployment config
├── pyrightconfig.json             # Type checking config
│
├── *.json state files:            # Runtime state persistence
│   ├── gemini_api_state.json      # Gemini key rotation state
│   ├── scraper_state.json         # Scraper progress state
│   ├── preprocess_state.json      # Preprocessor state
│   ├── processed_uris.json        # Scraped URI tracking
│   ├── binaural_scraper_state.json
│   ├── binaural_processed_uris.json
│   ├── audio_download_state.json
│   ├── user_profile.json
│   └── *.log files                # Processing logs
│
└── README.md                      # This file
```

---

## 🛠️ **Installation & Setup**

### **Prerequisites**

- Python 3.9.5 or higher
- Webcam access for emotion detection
- Stereo headphones (required for binaural beats)
- Stable internet connection
- Modern web browser (Chrome, Firefox, Safari, Edge)

### **Quick Start Guide**

#### **1. Clone the Repository**

```bash
git clone https://github.com/AetherSparks/Y.M.I.R.git
cd Y.M.I.R
```

#### **2. Create Virtual Environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

#### **4. Environment Setup**

Create a `.env` file in the root directory:

```env
SECRET_KEY=your-secret-key-here
EMAIL_USER=your-email@gmail.com
EMAIL_PASS=your-app-password
FLASK_ENV=development

# Spotify API (for music features & scraping)
SPOTIFY_CLIENT_ID=your-spotify-client-id
SPOTIFY_CLIENT_SECRET=your-spotify-client-secret

# Gemini API (for AI recommendations)
GEMINI_API_KEY=your-gemini-api-key
GEMINI_API_KEY2=your-backup-key
GEMINI_API_KEY3=another-backup-key

# Firebase (for authentication & storage)
GOOGLE_APPLICATION_CREDENTIALS=path/to/firebase-credentials.json
FIREBASE_PROJECT_ID=your-project-id

# YouTube Data API (for music streaming)
YOUTUBE_API_KEY=your-youtube-api-key
```

#### **5. Initialize Database**

```bash
python -c "from app import db; db.create_all()"
```

#### **6. Launch Application**

```bash
# Option A: Start all microservices via orchestrator
python start_ymir.py

# Option B: Start main app directly
python app.py
```

#### **7. Access the Application**

Open your browser and navigate to: `http://127.0.0.1:5000`

| Service | URL |
|---------|-----|
| Main App | http://localhost:5000 |
| AI Dashboard | http://localhost:5000/ai_app |
| Brainwave Lab | http://localhost:5000/experimental |
| Gaming Hub | http://localhost:5000/gaming |
| Therapist Finder | http://localhost:5000/therapist_finder |
| Face Microservice | http://localhost:5002 |
| Text Microservice | http://localhost:5003 |

---

## 📖 **Usage Guide**

### **Getting Started**

1. **Home Page**: Navigate to the landing page to understand Y.M.I.R's capabilities
2. **Emotion Detection**: Click "Start AI Analysis" to begin real-time emotion detection
3. **Grant Permissions**: Allow webcam access for visual emotion analysis
4. **Interact**: Use the chatbot for text-based emotional analysis
5. **Receive Recommendations**: Get personalized music and wellness suggestions

### **Core Modules**

#### **🎵 Music Therapy**
- Access personalized music recommendations based on your emotional state
- Multiple recommendation strategies: therapeutic, mood matching, mood regulation, adaptive
- Create and save favorite playlists
- Explore different music moods and genres across 3,200+ tracks
- Stream directly from integrated YouTube/SoundCloud platforms

#### **🧘 Meditation & Mindfulness**
- Choose from guided meditation sessions based on mood (anxious, stressed, tired, happy)
- Track meditation progress and streaks
- Access breathing exercises for immediate stress relief
- Explore sound therapy options with ambient soundscapes
- Use AI-powered journaling for emotional pattern recognition

#### **🧠 Brainwave Entrainment Lab**
- Select your wellness goal: Deep Sleep, Focus, Anxiety Relief, Relaxation, Stress Reduction, or Creative Flow
- Choose binaural beats (requires headphones) or isochronic tones (works with speakers)
- Each session generates unique audio with random variations for personalized experience
- 6 brainwave bands target different mental states (Delta through Gamma)
- Anonymized data collection helps improve the system

#### **🎮 AI Gaming Hub**
- Interact with NOVA, the AI NPC game recommendation assistant
- Receive game recommendations based on your emotional state
- Explore 4 story themes for AI-generated narratives
- Experience a 3D visual environment with HUD elements
- Track your gaming style, difficulty preference, and history

#### **📝 Digital Journaling**
- Write daily emotional entries with AI-powered sentiment analysis
- Receive insights about your emotional patterns and triggers
- Track mood trends over time with visual analytics
- Set and monitor personal wellness goals

#### **👥 Community Support**
- Connect with others on similar wellness journeys
- Share experiences in a safe, moderated environment
- Access professional support when needed
- Participate in group wellness challenges

#### **📊 Analytics Dashboard**
- View detailed emotion timelines with real-time webcam tracking
- Understand mood transition patterns
- Track wellness goal progress with streak monitoring
- Generate insights for personal growth

#### **🌍 Professional Therapist Finder**
- **Automatic Geolocation**: Browser-based location detection with user permission
- **Global Database Integration**: US (NPI Registry), India (Practo, Government portals), International platforms
- **Smart Location Parsing**: Supports "City, State", ZIP codes, and international addresses
- **Real-time API Connections**: Live data from healthcare platforms and government directories
- **Professional SVG Design**: Modern glassmorphism UI with custom icon system
- **Crisis Resources**: Immediate access to mental health helplines and emergency support
- **Verified Providers**: Licensed professionals with credential verification

---

## 🎯 **API Endpoints**

The main Flask application (`app.py`) exposes 50+ routes organized by category. Below is the complete list of all endpoints.

### **Page Routes**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing / home page |
| `/ai_app` | GET | Main AI emotion detection dashboard |
| `/about` | GET | About page |
| `/features` | GET | Feature showcase |
| `/pricing` | GET | Pricing information |
| `/privacy` | GET | Privacy policy |
| `/services` | GET | Service descriptions |
| `/contact` | GET/POST | Contact & support form |
| `/cookies` | GET | Cookie policy |
| `/image_processing` | GET | Advanced image analysis interface |
| `/therapist_finder` | GET | Professional therapist directory |

### **Emotion Detection & Camera APIs**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/video_feed` | GET | Real-time video stream for emotion detection |
| `/api/start_camera` | POST | Start webcam capture |
| `/api/stop_camera` | POST | Stop webcam capture |
| `/api/camera/start` | POST | Alternative camera start endpoint |
| `/api/camera/stop` | POST | Alternative camera stop endpoint |
| `/api/camera/settings` | POST | Update camera settings |
| `/api/emotions` | GET | Retrieve current facial emotion data |
| `/api/face_status` | GET | Face detection status |
| `/api/mediapipe/landmarks` | GET | MediaPipe face mesh landmarks data |
| `/api/combined_emotions` | GET | Fused facial + text emotion results |

### **Text & Chat APIs**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/text/analyze` | POST | Analyze text sentiment and emotions |
| `/api/text/chat` | POST | Send chat message to chatbot |
| `/api/text/conversation` | GET | Retrieve conversation history |
| `/api/text_status` | GET | Text analysis service status |

### **Music & Audio Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommend` | GET/POST | Get music recommendations |
| `/api/music/recommendations` | GET | API-based music recommendations |
| `/api/get_audio` | GET | Stream audio recommendations |
| `/api/get_album_art` | GET | Fetch album artwork |
| `/api/check_local_music` | GET | Check locally cached music |
| `/api/ai_recommend` | POST | Gemini AI-powered therapeutic recommendations |

### **Experimental (Brainwave Entrainment) Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/experimental` | GET | Brainwave entrainment lab home |
| `/experimental/binaural` | GET | Binaural beats session page |
| `/experimental/isochronic` | GET | Isochronic tones session page |
| `/experimental/binaural_tone/<filename>` | GET | Serve binaural beat audio files |
| `/experimental/isochronic_tone/<filename>` | GET | Serve isochronic tone audio files |
| `/experimental/foreground_music/<path:filename>` | GET | Serve foreground music tracks |

### **Gaming & Wellness Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/gaming` | GET | AI Gaming Hub interface |
| `/wellness` | GET | Comprehensive wellness tools |
| `/meditation` | GET | Meditation interface |
| `/meditation/result` | POST | Process meditation session |
| `/journal` | GET/POST | Journaling functionality |
| `/breathing` | GET/POST | Breathing exercises |
| `/goals` | GET/POST | Goal setting and tracking |
| `/check_goal/<int:goal_index>` | GET | Check off a specific goal |
| `/sound-therapy` | GET/POST | Sound therapy sessions |
| `/community` | GET/POST | Community forum |
| `/book-appointment` | POST | Book a therapy appointment |

### **Therapist Finder APIs**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/therapists` | GET/POST | Search for therapists with location support |
| `/api/geocode` | POST | Server-side geocoding for location detection |
| `/api/test_npi` | GET | Test NPI Registry API connection |

### **Learning & Feedback APIs**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/user_feedback` | POST | Submit emotion detection feedback |
| `/api/learning_analytics` | GET | Learning model performance analytics |
| `/api/emotion_suggestions` | GET | Get emotion improvement suggestions |

### **Miscellaneous**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/process_image` | POST | Upload and process an image for emotion analysis |
| `/health` | GET | Health check endpoint for load balancers |

### **Authentication (Firebase)**

*Routes added dynamically by `firebase_auth_backend.py`:*

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/login` | GET/POST | User login |
| `/signup` | GET/POST | User registration |
| `/logout` | GET | User logout |
| `/api/auth/verify` | POST | Verify Firebase token |

### **Standalone Microservice Endpoints**

These are available when running the individual microservice servers:

| Service | Endpoint | Port | Description |
|---------|----------|------|-------------|
| Face Microservice | `/` (GET) | 5002 | Face detection web interface |
| Face Microservice | `/video_feed` (GET) | 5002 | Face video stream |
| Face Microservice | `/api/emotions` (GET) | 5002 | Face emotion data |
| Face Microservice | `/api/analytics` (GET) | 5002 | Session analytics |
| Face Microservice | `/api/storage` (GET) | 5002 | Storage status |
| Face Microservice | `/api/export_session` (POST) | 5002 | Export session data |
| Text Microservice | `/` (GET) | 5003 | Chat web interface |
| Text Microservice | `/api/chat` (POST) | 5003 | Chat API |
| Text Microservice | `/api/status` (GET) | 5003 | Service status |
| Chatbot Web App | `/` (GET) | 5001 | Chatbot web interface |
| Chatbot Web App | `/api/chat` (POST) | 5001 | Chat API with Gemini streaming |
| Chatbot Web App | `/api/status` (GET) | 5001 | Service status |
| Face Web App | `/` (GET) | 5000 | Face detection standalone |
| Face Web App | `/video_feed` (GET) | 5000 | Video stream |
| Face Web App | `/api/start_camera` (POST) | 5000 | Start camera |
| Face Web App | `/api/stop_camera` (POST) | 5000 | Stop camera |
| Face Web App | `/api/emotions` (GET) | 5000 | Emotion data |
| Face Web App | `/api/analytics` (GET) | 5000 | Session analytics |
| Face Web App | `/api/storage` (GET) | 5000 | Firebase storage status |
| Face Web App | `/api/export_session` (POST) | 5000 | Export session |
| Face Web App | `/api/status` (GET) | 5000 | Service status |
| Face Web App | `/start_camera` (GET) | 5000 | Simple start page |

---

## 🗄️ **Database Schema**

### **Local Models (SQLite)**

```python
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    favorites = db.relationship('FavoriteSong', backref='user', lazy=True)

class FavoriteSong(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(120), nullable=False)
    artist = db.Column(db.String(120), nullable=False)
    link = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
```

### **Firebase Firestore Schema (Production)**

```
/emotion_sessions/{session_id}
  ├── user_id: string
  ├── created_at: timestamp
  ├── device_info: map
  └── readings/ (subcollection)
       ├── {reading_id}
       │   ├── timestamp: timestamp
       │   ├── emotions: map (dominant + all_emotions)
       │   ├── confidence: number
       │   ├── quality_score: number
       │   ├── environment_context: map
       │   ├── objects_detected: array
       │   └── session_id: string

/user_profiles/{user_id}
  ├── email: string
  ├── display_name: string
  ├── preferences: map
  ├── emotion_history: array
  ├── wellness_goals: array
  └── created_at: timestamp
```

---

## 📊 **Datasets**

### **1. Original Music Dataset** (`Y.M.I.R. original dataset.csv`)
- **3,465 tracks** scraped from Spotify
- **Features**: Track URI, Name, Artist, Album, Popularity, Danceability, Energy, Key, Loudness, Mode, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Duration

### **2. Therapeutic Music Enriched** (`therapeutic_music_enriched.csv`)
- **3,213 tracks** with enhanced emotion labeling
- **Additional fields**: Mood_Label (10 categories), Confidence score, Mental_Health_Benefit (4 per track), Musical_Features description
- **Emotion categories**: Joy, Sadness, Excitement, Calm, Anger, Fear, Surprise, Nostalgia, Love, Neutral

### **3. Binaural Beats Experimental** (`binaural_beats_experimental.csv`)
- **10,000+ tracks** scraped for brainwave entrainment
- **Categories**: Experimental_Category (focus, sleep, meditation), Brainwave_Type (Beta, Alpha, Delta, Theta, Gamma), Purpose
- **Source**: Specialized Spotify playlists for binaural beats, isochronic tones, and healing frequencies

### **Sample Dataset Structure**

```csv
# Therapeutic Music
Track Name,Artist Name,Danceability,Energy,Valence,Tempo,Mood_Label,Mental_Health_Benefit
Jo Tum Mere Ho,Anuv Jain,0.46,0.302,0.176,123.871,Sadness,"Mood Upliftment, Emotional Release"
Choo Lo,The Local Train,0.512,0.695,0.351,145.956,Joy,"Energy Boost, Increased Motivation"

# Binaural Beats
Track URI,Track Name,Brainwave_Type,Purpose,Danceability,Energy,Valence,Tempo
spotify:track:...,40 Hz Binaural Beats,Beta (13-30Hz),focus,0.1,0.2,0.4,60.0
spotify:track:...,3 Hz Deep Sleep,Delta (0.5-4Hz),sleep,0.1,0.2,0.4,60.0
```

---

## 🚀 **Deployment**

### **Vercel Deployment**

Y.M.I.R is configured for serverless deployment on Vercel:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "api/index.py"
    }
  ]
}
```

### **Local Development**

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy to Vercel
vercel --prod
```

### **Microservices Deployment**

```bash
# Start individual services
python text_microservice_working.py   # Port 5003
python face_microservice_working.py   # Port 5002
python chatbotwebapp.py               # Port 5004

# Or use the orchestrator
python start_ymir.py
```

### **Environment Variables for Production**

- `SECRET_KEY`: Flask secret key
- `EMAIL_USER`: SMTP email address
- `EMAIL_PASS`: SMTP password
- `DATABASE_URL`: Production database URL
- `GEMINI_API_KEY` through `GEMINI_API_KEY8`: Gemini API keys for rotation
- `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET`: Spotify API access
- `GOOGLE_APPLICATION_CREDENTIALS`: Firebase service account
- `FIREBASE_PROJECT_ID`: Firebase project identifier

---

## 🧪 **Testing**

### **Manual Testing Checklist**

- [ ] Webcam emotion detection functionality with face quality assessment
- [ ] Multi-face detection (up to 3 faces)
- [ ] Chatbot emotion analysis with streaming responses
- [ ] Music recommendation accuracy across 4 strategies
- [ ] User favorites system with CRUD operations
- [ ] Meditation timer with mood-based scripts
- [ ] Journal entry processing with sentiment analysis
- [ ] Community post creation with anonymous option
- [ ] Binaural beats audio generation and playback
- [ ] Isochronic tones audio generation and playback
- [ ] Gaming hub NPC interaction
- [ ] Therapist finder geolocation search
- [ ] Email notification system
- [ ] Firebase authentication flow

### **Automated Testing**

```bash
# Run multimodal system tests
cd tests
python test_multimodal_system.py

# Test facial-firebase integration
python test_facial_firebase.py

# Test recommendation engine
python enhancements/src-new/recommendation/test_recommendations.py

# Test Gemini API rotation
python -c "from gemini_api_manager import GeminiAPIManager; m = GeminiAPIManager(); print(f'Active keys: {sum(1 for k in m.api_keys if k.is_active)}')"
```

### **Browser Compatibility**

- ✅ Chrome (Recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge
- ⚠️ Internet Explorer (Limited support)

---

## 🤝 **Contributing**

We welcome contributions from developers, researchers, and mental health professionals!

### **How to Contribute**

1. **Fork the Repository**

   ```bash
   git clone https://github.com/yourusername/Y.M.I.R.git
   ```

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make Your Changes**
   - Follow PEP 8 style guidelines
   - Add proper documentation
   - Include tests for new features

4. **Commit Your Changes**

   ```bash
   git commit -m 'Add amazing feature'
   ```

5. **Push to Your Branch**

   ```bash
   git push origin feature/amazing-feature
   ```

6. **Open a Pull Request**
   - Describe your changes in detail
   - Include screenshots for UI changes
   - Reference any related issues

### **Contribution Guidelines**

- 🔒 **Security**: Never commit sensitive data or API keys
- 📝 **Documentation**: Update README and code comments
- 🧪 **Testing**: Include tests for new functionality
- 🎨 **UI/UX**: Maintain consistent design language with glassmorphism effects
- 🧠 **AI Ethics**: Consider bias and fairness in ML implementations
- 🌍 **International Support**: Ensure features work across different countries and healthcare systems

---

## 🛡️ **Privacy & Security**

### **Data Protection**

- All emotion data is processed locally when possible
- Face detection runs on-device; raw images never transmitted
- User data is encrypted in transit and at rest
- Firebase Firestore with session-based isolation ensures GDPR compliance
- Users can delete their data at any time
- Chat conversations encrypted with cryptography library

### **Ethical AI**

- Transparent emotion detection algorithms with confidence scoring
- No discrimination based on demographic characteristics
- User consent required for all data collection with privacy mode toggle
- Regular bias auditing of ML models
- Face quality assessment prevents low-quality/false-positive detection

### **Security Measures**

- Input sanitization across all endpoints (XSS prevention)
- Environment-variable-based secrets (no hardcoded credentials)
- Firebase security rules for database access control
- CORS configuration for microservice isolation
- SMTP encryption for email communications

---

## 🔮 **Future Scope & Roadmap**

### **Phase 1: Core Enhancement** (Current — Q3 2025)

- [x] Multi-modal emotion detection (facial + text + fusion)
- [x] Ensemble ML pipeline with 8 models
- [x] Gemini AI integration with automatic key rotation
- [x] Microservices architecture extraction
- [x] Binaural beats and isochronic tone generation
- [x] AI Gaming Hub with NPC interaction
- [x] Firebase authentication and Firestore storage
- [x] Production music scraper with state persistence
- [x] Therapist finder with global geolocation
- [x] Comprehensive wellness toolkit

### **Phase 2: Platform Expansion** (Q3–Q4 2025)

- [ ] **Advanced ML Model Serving**: Containerized model deployment with GPU support
- [ ] **Real-time Collaborative Features**: Multi-user emotion sessions with WebSocket support
- [ ] **Mobile Application**: React Native / Flutter cross-platform mobile app
- [ ] **Enhanced Privacy Controls**: End-to-end encryption for all communications
- [ ] **Voice Emotion Analysis**: Speech tone and pitch analysis for emotion detection
- [ ] **Biometric Data Fusion**: Heart rate, skin conductance, and wearable device integration
- [ ] **Professional Therapist Portal**: Dashboard for therapists to monitor patient progress

### **Phase 3: Ecosystem Development** (Q1–Q2 2026)

- [ ] **Third-Party Integrations**: Spotify, Apple Music, Google Play Music native APIs
- [ ] **Clinical Trial Partnerships**: Research studies for clinical validation
- [ ] **Research Publication Support**: Open-source benchmark dataset release
- [ ] **Advanced Analytics Dashboard**: AI-powered insights and predictive mood modeling
- [ ] **Multi-Language Support**: NLP models for Hindi, Spanish, French, Arabic, Mandarin
- [ ] **Brainwave Entrainment Enhancement**: Real-time EEG biofeedback integration
- [ ] **Gaming Expansion**: VR/AR therapeutic gaming experiences

### **Phase 4: Enterprise & Scale** (Q3 2026+)

- [ ] **Kubernetes Deployment**: Auto-scaling container orchestration
- [ ] **Distributed Processing**: Message queues (Kafka/RabbitMQ) for async processing
- [ ] **Service Mesh**: Istio for microservice observability and traffic management
- [ ] **Revenue Model**: Freemium with premium therapeutic features
- [ ] **Enterprise API**: White-label emotion detection API for B2B customers
- [ ] **Global Partnerships**: Integration with national healthcare systems
- [ ] **Clinical Certification**: FDA/CE marking for medical-grade wellness tool

---

## 🏆 **Recognition & Awards**

- 🥇 **Best AI Innovation** — College Technical Symposium 2024
- 🌟 **Mental Health Tech Excellence** — University Research Fair
- 💡 **Creative Solution Award** — DTI & AI Project Competition
- 🎯 **Selected for Smart India Hackathon 2025**

---

## 📚 **Academic References**

1. **Music Therapy Research**: Effects of music on emotional regulation (Johnson et al., 2023)
2. **Emotion Recognition**: Deep learning approaches to facial emotion detection (Smith et al., 2023)
3. **Brainwave Entrainment**: Binaural beats and cognitive performance enhancement (Garcia et al., 2022)
4. **Ensemble Learning**: Combining classifiers for improved emotion recognition (Patel et al., 2024)
5. **Mental Health Technology**: Digital interventions for anxiety and depression (Brown et al., 2024)
6. **AI Ethics**: Responsible AI in healthcare applications (Davis et al., 2023)
7. **Multimodal Fusion**: Integrating visual and textual signals for affective computing (Zhang et al., 2023)

---

## 👨‍💻 **Development Team**

<div align="center">

| **Abhiraj Ghose** | **Pallav Sharma** |
|:---:|:---:|
| E23CSEU0014 | E23CSEU0022 |
| **Full-Stack Developer & AI Specialist** | **Full-Stack Developer & UI/UX Designer** |
| [![GitHub](https://img.shields.io/badge/GitHub-AetherSparks-blue?logo=github)](https://github.com/AetherSparks) | [![GitHub](https://img.shields.io/badge/GitHub-pallav110-blue?logo=github)](https://github.com/pallav110) |

</div>

### **Contributions Breakdown**

#### **Abhiraj Ghose** — Emerging Areas: Novelty Features, Systems Architecture & ML Engineering
- **Brainwave Entrainment Lab Generation** (Novelty — End-to-End Flow): Complete experimental pipeline from concept to production — binaural beat waveform synthesis (left/right ear frequency offset), isochronic tone generation (amplitude-modulated pulses), 6 brainwave band classification (Delta/Theta/Alpha/Beta/Gamma/Epsilon), audio parameter randomization with amplitude modulation and fade envelopes, goal-based session selection (Sleep/Focus/Anxiety/Relaxation/Stress/Creativity), WAV file serving via Flask routes, synthetic audio generation engine (`binaural_audio_testing.py`, `isochronic_audio_testing.py`), 36 generated audio files
- **Binaural Beats Data Pipeline** (Novelty): Spotify API scraper for 10,000+ brainwave entrainment tracks (`binaural_beats_scraper.py`), audio downloader with preview clip extraction (`binaural_audio_downloader.py`), dataset categorization by brainwave type and therapeutic purpose, incremental scraping with state persistence, dual scraper/downloader state machine
- **AI Emotion Dashboard**: Complete 10,681-line real-time emotion detection interface with webcam integration, emotion smoothing, confidence visualization, responsive glassmorphism + cyberpunk design, real-time analytics, multi-face tracking display
- **Microservices Architecture**: Complete extraction from 2,071-line monolith into 7 focused service modules + 3 independently deployable microservice servers (`face_microservice_working.py` port 5002, `text_microservice_working.py` port 5003, Independent web apps), system orchestrator with health checks and CORS inter-service communication (`start_ymir.py`), service-level scaling and isolation
- **Ensemble Machine Learning Pipeline**: MLEmotionDetector with 6+ models (RandomForest, GradientBoosting, SVC, MLPClassifier, XGBoost, LightGBM), VotingClassifier/StackingClassifier ensemble with soft voting, SMOTE oversampling for class imbalance, StratifiedKFold 5-fold cross-validation, fixed overfitting from 97% → realistic 65-75% accuracy, 14 engineered audio features, synthetic training dataset generation based on music theory research
- **Text Emotion Detection**: Multiple transformer models (DistilBERT, RoBERTa) with ensemble voting, streaming chatbot responses like ChatGPT, conversation history management with encryption (`cryptography.fernet`), context-aware processing, Gemini-powered therapeutic responses with fallback chatbot
- **Production Music Recommender**: Full ML training pipeline with feature engineering, data leakage elimination, class imbalance regularization, production-ready evaluation framework with cross-validation reporting
- **Firebase Integration**: Authentication backend (email/password + Google OAuth), Firestore schema design for emotion sessions and user profiles, real-time data sync, offline resilience with local buffer, batch processing (5 readings/batch, 3s sync interval)
- **UI/UX & Visual Design System**: Complete glassmorphism + cyberpunk aesthetic, `THEME_MANAGE.md` (789 lines) documenting color palettes, typography, component library, animation framework (GSAP, CSS, Three.js), responsive breakpoints, dark/light theme overrides

#### **Pallav Sharma** — Core Platform, UI/UX, Feature Integration & Deployment
- **Facial Emotion Detection System**: fer_enhanced_v3.py architecture — MediaPipe face mesh/pose/hand tracking, YOLO environmental context (YOLOv11/v10/v9/v8 variants with `EmotionContextAnalyzer`), DeepFace ensemble (VGG-Face/FaceNet/Default backends with thread-safe timeout protection), face quality assessment (POOR/FAIR/GOOD/EXCELLENT), multi-face tracking (up to 3 faces), emotion smoothing with temporal stability window, privacy mode toggle
- **Gemini API Integration**: Automatic key rotation manager supporting 8+ API keys, quota exhaustion detection and recovery, 24-hour reset window, persistent state management (`gemini_api_state.json`), seamless failover with zero downtime
- **True ML Emotion Context**: Unsupervised learning system for emotion-environment pattern discovery using KMeans clustering (8 environment + 6 emotion clusters), MLPRegressor neural network (100-50-25 layers), PCA dimensionality reduction (50 components), TF-IDF vectorization (200 features, 1-2 n-grams), blended inference (80/20 DeepFace-to-ML ratio based on confidence), model persistence via pickle, requirement of 20+ observations before training
- **Multimodal Fusion Engine**: 5 fusion strategies (simple, adaptive, confidence-based, temporal weighted, weighted average), `unified_emotion_music_system.py` (1,911 lines) reading from Firebase + JSON data sources, `real_emotion_combiner.py` for real-time combination, session management with analytics, music recommendation integration across 4 strategies (therapeutic, mood_matching, mood_regulation, adaptive)
- **Wellness Tools Suite**: Meditation module with mood-based script selection (anxious/stressed/tired/happy), breathing exercises with adaptive techniques, AI-powered journaling with VADER sentiment analysis, goal tracking with streak monitoring, sound therapy with ambient soundscapes, movie recommendations
- **Therapist Finder**: Global geolocation search (OpenStreetMap Nominatim), NPI Registry API (US), Practo API (India), government health portal integration, crisis helpline resources, professional SVG glassmorphism UI
- **AI Gaming Hub**: Three.js 3D particle environment, NOVA NPC chatbot (pure frontend JavaScript with personality-driven dialogue), emotion-based game recommendations, 4 story themes (Fantasy/Sci-Fi/Cyberpunk/Medieval), 4 emotion-based styles (Excited/Calm/Focused/Competitive)
- **Deployment & Infrastructure**: Vercel serverless configuration, multipage Flask routing across 50+ endpoints, image processing module (edge detection: Sobel/Canny/Laplacian, histogram equalization, face detection), environment variable management, comprehensive documentation suite (12 DOCS files)
---

## 🙏 **Acknowledgments**

### **Open Source Libraries**

- [DeepFace](https://github.com/serengil/deepface) — Facial emotion detection framework
- [Transformers](https://huggingface.co/transformers/) — State-of-the-art NLP models by HuggingFace
- [Flask](https://flask.palletsprojects.com/) — Lightweight web framework
- [OpenCV](https://opencv.org/) — Computer vision library
- [MediaPipe](https://mediapipe.dev/) — Real-time face and hand tracking
- [scikit-learn](https://scikit-learn.org/) — Machine learning framework
- [XGBoost](https://xgboost.readthedocs.io/) — Gradient boosting framework
- [LightGBM](https://lightgbm.readthedocs.io/) — Gradient boosting framework
- [Firebase](https://firebase.google.com/) — Authentication and cloud storage
- [Google Generative AI](https://ai.google.dev/) — Gemini API for AI recommendations
- [Spotipy](https://spotipy.readthedocs.io/) — Spotify Web API wrapper
- [Three.js](https://threejs.org/) — 3D graphics for gaming hub

### **Research & Inspiration**

- Mental health research community
- Open-source AI/ML community
- Music therapy practitioners
- Brainwave entrainment researchers
- Beta testers and early adopters

### **Special Thanks**

- University faculty and mentors
- Mental health professionals who provided guidance
- Friends and family who supported development
- The open-source community for invaluable tools and libraries

---

## 📄 **License**

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Y.M.I.R Development Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🆘 **Support & Contact**

### **Getting Help**

- 📧 **Email**: support@ymir-ai.com
- 💬 **Discord**: [Y.M.I.R Community](https://discord.gg/ymir-ai)
- 🐛 **Issues**: [GitHub Issues](https://github.com/AetherSparks/Y.M.I.R/issues)
- 📖 **Documentation**: [Wiki](https://github.com/AetherSparks/Y.M.I.R/wiki)

### **Crisis Support**

If you're experiencing a mental health crisis, please reach out to:

- **National Suicide Prevention Lifeline**: 988 (US)
- **Crisis Text Line**: Text HOME to 741741
- **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/

---

<div align="center">

### **⭐ Star this repository if Y.M.I.R helped you! ⭐**

**Made with ❤️ for mental health and wellness**

---

_"Technology should serve humanity's deepest needs — and there's no need deeper than mental well-being."_

---

[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=AetherSparks.Y.M.I.R)](https://github.com/AetherSparks/Y.M.I.R)
[![Stars](https://img.shields.io/github/stars/AetherSparks/Y.M.I.R?style=social)](https://github.com/AetherSparks/Y.M.I.R/stargazers)
[![Forks](https://img.shields.io/github/forks/AetherSparks/Y.M.I.R?style=social)](https://github.com/AetherSparks/Y.M.I.R/network/members)

</div>
