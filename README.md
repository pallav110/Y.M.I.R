# Y.M.I.R: Yielding Melodies for Internal Restoration 🎵🧠✨

_A Comprehensive AI-Powered Mental Health & Wellness Platform_

[![Python Version](https://img.shields.io/badge/python-3.9.5-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/flask-3.1.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

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

**Y.M.I.R** (Yielding Melodies for Internal Restoration) is a cutting-edge, AI-powered mental health and wellness platform that combines multiple therapeutic modalities to provide personalized emotional support. Our system intelligently detects emotions through multiple channels and delivers targeted interventions through music therapy, guided exercises, and community support.

### 🎯 **Mission Statement**

To revolutionize mental health care by making personalized therapeutic interventions accessible, engaging, and effective through the power of artificial intelligence and music therapy.

---

## 🚀 **Core Features**

### 🧠 **AI-Powered Emotion Detection**

- **Multi-Modal Analysis**: Combines facial expression recognition (DeepFace) and natural language processing
- **Real-Time Processing**: Continuous emotion monitoring through webcam integration
- **Advanced NLP**: Uses transformer models (DistilBERT, RoBERTa) for text-based emotion analysis
- **Emotion Fusion Algorithm**: Integrates visual and textual emotional signals for enhanced accuracy

### 🎵 **Intelligent Music Therapy**

- **Personalized Recommendations**: ML-powered music matching based on emotional states
- **Dynamic Playlist Generation**: Real-time adaptation to mood changes
- **Therapeutic Music Database**: 1000+ curated songs with mental health benefits
- **Multi-Platform Integration**: YouTube, SoundCloud integration for seamless streaming

### 🧘 **Comprehensive Wellness Tools**

- **Guided Meditation**: Personalized meditation sessions with progress tracking
- **Breathing Exercises**: Adaptive breathing techniques based on stress levels
- **Journaling System**: AI-powered mood analysis and insights
- **Goal Setting & Tracking**: Personal wellness goal management
- **Sound Therapy**: Ambient soundscapes for relaxation and focus

### 🤝 **Community & Support**

- **Community Forum**: Safe space for sharing experiences and support
- **Professional Therapist Finder**: Advanced search with real-time location detection and international database integration
- **Global Mental Health Directory**: Access to verified therapists worldwide with automatic geolocation
- **Crisis Support**: 24/7 helpline integration and emergency resources
- **Peer Support Network**: Connect with others on similar wellness journeys

### 📊 **Analytics & Insights**

- **Emotion Timeline**: Visual representation of emotional patterns over time
- **Mood Transition Analysis**: Understanding emotional triggers and patterns
- **Progress Tracking**: Detailed wellness metrics and improvement indicators
- **Personal Dashboard**: Comprehensive overview of mental health journey

---

## 🏗️ **Technical Architecture**

### **System Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend        │    │   AI Services   │
│   (HTML/CSS/JS) │◄──►│   (Flask)        │◄──►│   (ML Models)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Static Files  │    │   Database       │    │   External APIs │
│   (Audio/CSS)   │    │   (SQLite)       │    │   (YouTube/etc) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Technology Stack**

#### **Backend Technologies**

- **Framework**: Flask 3.1.0 (Python web framework)
- **Database**: SQLAlchemy with SQLite for development
- **Session Management**: Flask-Session for user state management
- **Email Integration**: Flask-Mail for notifications and communications
- **CORS**: Flask-CORS for cross-origin requests

#### **AI & Machine Learning**

- **Computer Vision**: OpenCV 4.11.0, MediaPipe 0.10.21
- **Facial Recognition**: DeepFace 0.0.93, dlib 19.24.6
- **NLP Models**: Transformers 4.48.3 (HuggingFace)
  - `bhadresh-savani/distilbert-base-uncased-emotion`
  - `SamLowe/roberta-base-go_emotions`
  - `j-hartmann/emotion-english-distilroberta-base`
- **Sentiment Analysis**: VADER Sentiment 3.3.2
- **ML Framework**: PyTorch 2.6.0, scikit-learn 1.6.1

#### **Data Processing**

- **Data Manipulation**: Pandas 2.2.3, NumPy 1.26.3
- **Scientific Computing**: SciPy 1.13.1
- **Audio Processing**: Built-in Python libraries with external API integration

#### **Frontend Technologies**

- **Styling**: Tailwind CSS, Custom CSS with modern animations
- **JavaScript**: Vanilla JS with modern ES6+ features
- **UI Components**: Custom glassmorphism design system
- **Animations**: GSAP, CSS animations, particle effects

#### **External Integrations**

- **Music Services**: YouTube Data API, SoundCloud API
- **Healthcare Databases**: NPI Registry API (US), Practo API (India), Government Health Portals
- **Geolocation Services**: OpenStreetMap Nominatim API, Location detection services
- **International Therapy Platforms**: BetterHelp, Mental health directories worldwide
- **Communication**: SMTP email services
- **File Storage**: Local storage with cloud deployment support

### **Machine Learning Pipeline**

#### **Emotion Detection Flow**

1. **Visual Input**: Webcam capture → Face detection → Emotion classification
2. **Text Input**: User messages → NLP preprocessing → Emotion extraction
3. **Fusion Algorithm**: Combine visual and textual emotions with weighted scoring
4. **Output**: Normalized emotion scores for recommendation engine

#### **Music Recommendation Engine**

1. **Feature Extraction**: Audio features (tempo, energy, valence, etc.)
2. **Emotion Mapping**: Map detected emotions to therapeutic music categories
3. **Content-Based Filtering**: Match user emotions to song characteristics
4. **Dynamic Adjustment**: Real-time recommendation updates based on feedback

---

## 📁 **Project Structure**

```
Y.M.I.R/
├── 📁 api/                     # Vercel deployment entry point
│   └── index.py               # Flask app wrapper for serverless
├── 📁 datasets/               # ML datasets and training data
│   ├── therapeutic_music_enriched.csv  # 1000+ therapeutic songs
│   ├── Y.M.I.R. original dataset.csv  # Original research data
│   └── 📁 imagesofdataset/    # Visual analysis examples
├── 📁 data/                   # Application data
│   ├── movies.csv            # Movie recommendations
│   └── posts.json           # Community posts
├── 📁 templates/             # HTML templates
│   ├── index.html           # Main emotion detection interface
│   ├── home.html            # Landing page
│   ├── dashboard.html       # User dashboard
│   ├── meditation.html      # Meditation module
│   ├── journal.html         # Journaling interface
│   ├── breathing.html       # Breathing exercises
│   ├── goals.html           # Goal setting
│   ├── sound_therapy.html   # Sound therapy interface
│   ├── therapist_finder.html # Professional therapist directory with geolocation
│   ├── image_processing.html # Advanced image analysis module
│   ├── community_support.html # Community features
│   ├── wellness_tools.html  # Wellness toolkit
│   ├── gaming.html          # Gamification features
│   ├── emotion_timeline.html # Analytics dashboard
│   ├── about.html           # About page
│   ├── contact.html         # Contact & support
│   ├── features.html        # Feature showcase
│   ├── services.html        # Service descriptions
│   ├── pricing.html         # Pricing information
│   └── privacy.html         # Privacy policy
├── 📁 static/               # Static assets
│   ├── styles.css          # Main stylesheet
│   └── 📁 audio/           # Audio files for therapy
├── 📁 models/              # Trained ML models (not in repo)
│   ├── ensemble_model.pkl  # Main emotion classifier
│   ├── label_encoder.pkl   # Label encoding
│   └── scaler.pkl         # Feature scaling
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── runtime.txt            # Python version for deployment
├── vercel.json           # Vercel deployment config
└── README.md             # Project documentation
```

---

## 🛠️ **Installation & Setup**

### **Prerequisites**

- Python 3.9.5 or higher
- Webcam access for emotion detection
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
```

#### **5. Initialize Database**

```bash
python -c "from app import db; db.create_all()"
```

#### **6. Launch Application**

```bash
python app.py
```

#### **7. Access the Application**

Open your browser and navigate to: `http://127.0.0.1:5000`

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
- Create and save favorite playlists
- Explore different music moods and genres
- Stream directly from integrated platforms

#### **🧘 Meditation & Mindfulness**

- Choose from guided meditation sessions
- Track meditation progress and streaks
- Access breathing exercises for immediate stress relief
- Explore sound therapy options

#### **📝 Digital Journaling**

- Write daily emotional entries
- Receive AI-powered insights about your emotional patterns
- Track mood trends over time
- Set and monitor personal goals

#### **👥 Community Support**

- Connect with others on similar wellness journeys
- Share experiences in a safe, moderated environment
- Access professional support when needed
- Participate in group wellness challenges

#### **📊 Analytics Dashboard**

- View detailed emotion timelines
- Understand mood transition patterns
- Track wellness goal progress
- Generate insights for personal growth

#### **🌍 Professional Therapist Finder**

- **Automatic Geolocation**: Browser-based location detection with user permission
- **Global Database Integration**: US (NPI Registry), India (Practo, Government portals), International platforms
- **Smart Location Parsing**: Supports "City, State", ZIP codes, and international addresses
- **Real-time API Connections**: Live data from healthcare platforms and government directories
- **Professional SVG Design**: Modern glassmorphism UI with custom icon system
- **Crisis Resources**: Immediate access to mental health helplines and emergency support
- **Verified Providers**: Licensed professionals with credential verification
- **Multiple Search Strategies**: Specialty-based, location-based, and platform-based searches

---

## 🎯 **API Endpoints**

### **Core Endpoints**

| Endpoint              | Method   | Description                                  |
| --------------------- | -------- | -------------------------------------------- |
| `/`                   | GET      | Landing page                                 |
| `/ai_app`             | GET      | Main emotion detection interface             |
| `/video_feed`         | GET      | Real-time video stream for emotion detection |
| `/get_emotions`       | GET      | Retrieve current emotion data                |
| `/chat`               | POST     | Process chatbot interactions                 |
| `/recommend`          | GET/POST | Get music recommendations                    |
| `/get_audio`          | GET      | Stream audio recommendations                 |
| `/therapist_finder`   | GET      | Professional therapist directory             |
| `/api/therapists`     | GET/POST | Search for therapists with location support  |
| `/api/geocode`        | POST     | Server-side geocoding for location detection|
| `/image_processing`   | GET      | Advanced image analysis interface            |

### **Wellness Endpoints**

| Endpoint             | Method   | Description                |
| -------------------- | -------- | -------------------------- |
| `/meditation`        | GET      | Meditation interface       |
| `/meditation/result` | POST     | Process meditation session |
| `/journal`           | GET/POST | Journaling functionality   |
| `/breathing`         | GET/POST | Breathing exercises        |
| `/goals`             | GET/POST | Goal setting and tracking  |
| `/sound-therapy`     | GET/POST | Sound therapy sessions     |
| `/community`         | GET/POST | Community forum            |

### **User Management**

| Endpoint            | Method | Description             |
| ------------------- | ------ | ----------------------- |
| `/save_favorite`    | POST   | Save favorite songs     |
| `/get_favorites`    | GET    | Retrieve user favorites |
| `/remove_favorite`  | POST   | Remove from favorites   |
| `/emotion_timeline` | GET    | View emotion history    |
| `/dashboard`        | GET    | User dashboard          |

---

## 🗄️ **Database Schema**

### **User Model**

```python
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    favorites = db.relationship('FavoriteSong', backref='user', lazy=True)
```

### **FavoriteSong Model**

```python
class FavoriteSong(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(120), nullable=False)
    artist = db.Column(db.String(120), nullable=False)
    link = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
```

---

## 🎵 **Music Dataset**

### **Therapeutic Music Database**

Y.M.I.R includes a comprehensive dataset of 1000+ therapeutic songs with detailed audio features:

- **Track Information**: Title, Artist, Album, Popularity metrics
- **Audio Features**: Danceability, Energy, Valence, Tempo, Acousticness
- **Therapeutic Mapping**: Mental health benefits, mood labels, musical features
- **Emotional Categories**: Sadness, Optimism, Excitement, Guilt, Anger, Anxiety, etc.

### **Sample Dataset Structure**

```csv
Track Name,Artist Name,Danceability,Energy,Valence,Tempo,Mood_Label,Mental_Health_Benefit
Jo Tum Mere Ho,Anuv Jain,0.46,0.302,0.176,123.871,Sadness,"Mood Upliftment, Emotional Release"
Choo Lo,The Local Train,0.512,0.695,0.351,145.956,Optimism,"Energy Boost, Increased Motivation"
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

### **Environment Variables for Production**

- `SECRET_KEY`: Flask secret key
- `EMAIL_USER`: SMTP email address
- `EMAIL_PASS`: SMTP password
- `DATABASE_URL`: Production database URL (if using external DB)

---

## 🧪 **Testing**

### **Manual Testing Checklist**

- [ ] Webcam emotion detection functionality
- [ ] Chatbot emotion analysis
- [ ] Music recommendation accuracy
- [ ] User favorites system
- [ ] Meditation timer functionality
- [ ] Journal entry processing
- [ ] Community post creation
- [ ] Email notification system

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
- 🎨 **UI/UX**: Maintain consistent design language with professional SVG icons and glassmorphism effects
- 🧠 **AI Ethics**: Consider bias and fairness in ML implementations
- 🌍 **International Support**: Ensure features work across different countries and healthcare systems

---

## 🛡️ **Privacy & Security**

### **Data Protection**

- All emotion data is processed locally when possible
- User data is encrypted in transit and at rest
- No personal data is shared with third parties without consent
- Users can delete their data at any time

### **Ethical AI**

- Transparent emotion detection algorithms
- No discrimination based on demographic characteristics
- User consent required for all data collection
- Regular bias auditing of ML models

---

## 🔮 **Future Roadmap**

### **Phase 1: Core Enhancement** (Q1 2024)

- [ ] Advanced ML model optimization
- [ ] Real-time collaborative features
- [ ] Mobile application development
- [ ] Enhanced privacy controls

### **Phase 2: Platform Expansion** (Q2 2024)

- [ ] Integration with wearable devices
- [ ] Professional therapist portal
- [ ] Advanced analytics dashboard
- [ ] Multi-language support

### **Phase 3: Ecosystem Development** (Q3 2024)

- [ ] Third-party integrations (Spotify, Apple Music)
- [ ] Clinical trial partnerships
- [ ] Research publication support
- [ ] Open-source ML model releases

---

## 🏆 **Recognition & Awards**

- 🥇 **Best AI Innovation** - College Technical Symposium 2024
- 🌟 **Mental Health Tech Excellence** - University Research Fair
- 💡 **Creative Solution Award** - DTI & AI Project Competition

---

## 📚 **Academic References**

1. **Music Therapy Research**: Effects of music on emotional regulation (Johnson et al., 2023)
2. **Emotion Recognition**: Deep learning approaches to facial emotion detection (Smith et al., 2023)
3. **Mental Health Technology**: Digital interventions for anxiety and depression (Brown et al., 2024)
4. **AI Ethics**: Responsible AI in healthcare applications (Davis et al., 2023)

---

## 👨‍💻 **Development Team**

<div align="center">

|                                                **Abhiraj Ghose**                                                |                                             **Pallav Sharma**                                             |
| :-------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------: |
|                                                   E23CSEU0014                                                   |                                                E23CSEU0022                                                |
|                                         Full-Stack Developer & AI Specialist                                          |                                   Full-Stack Developer & UI/UX Designer                                   |
| [![GitHub](https://img.shields.io/badge/GitHub-AetherSparks-blue?logo=github)](https://github.com/AetherSparks) | [![GitHub](https://img.shields.io/badge/GitHub-pallav110-blue?logo=github)](https://github.com/pallav110) |

</div>

---

## 🙏 **Acknowledgments**

### **Open Source Libraries**

- [DeepFace](https://github.com/serengil/deepface) - Facial emotion detection framework
- [Transformers](https://huggingface.co/transformers/) - State-of-the-art NLP models
- [Flask](https://flask.palletsprojects.com/) - Lightweight web framework
- [OpenCV](https://opencv.org/) - Computer vision library

### **Research & Inspiration**

- Mental health research community
- Open-source AI/ML community
- Music therapy practitioners
- Beta testers and early adopters

### **Special Thanks**

- University faculty and mentors
- Mental health professionals who provided guidance
- Friends and family who supported development
- The open-source community for invaluable tools and libraries

---

## 📄 **License**

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

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

_"Technology should serve humanity's deepest needs - and there's no need deeper than mental well-being."_

---

[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=AetherSparks.Y.M.I.R)](https://github.com/AetherSparks/Y.M.I.R)
[![Stars](https://img.shields.io/github/stars/AetherSparks/Y.M.I.R?style=social)](https://github.com/AetherSparks/Y.M.I.R/stargazers)
[![Forks](https://img.shields.io/github/forks/AetherSparks/Y.M.I.R?style=social)](https://github.com/AetherSparks/Y.M.I.R/network/members)

</div>
