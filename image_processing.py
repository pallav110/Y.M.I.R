"""
Y.M.I.R Image Processing Module
Advanced image processing techniques for enhanced emotion detection
"""

import cv2
import numpy as np
from scipy import ndimage
import base64
from io import BytesIO
from PIL import Image

class ImageProcessor:
    def __init__(self):
        """Initialize image processing with pre-loaded models"""
        try:
            # Load Haar Cascade for face detection (Viola-Jones method)
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            # Check if cascades loaded successfully
            if self.face_cascade.empty():
                print("⚠️ Warning: Face cascade failed to load")
                self.face_cascade = None
            if self.eye_cascade.empty():
                print("⚠️ Warning: Eye cascade failed to load")
                self.eye_cascade = None
                
        except Exception as e:
            print(f"⚠️ Warning: Error initializing cascades: {e}")
            self.face_cascade = None
            self.eye_cascade = None
        
    def edge_detection(self, image, method='canny'):
        """
        Apply edge detection algorithms for enhanced emotion analysis
        Methods: sobel, canny, laplacian
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        if method == 'sobel':
            # Sobel edge detection
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = np.uint8(edges / edges.max() * 255)
            
        elif method == 'canny':
            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
        elif method == 'laplacian':
            # Laplacian edge detection
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
            
        return edges
    
    def color_model_conversion(self, image, target_model='HSV'):
        """
        Convert between color models for enhanced facial analysis
        Supported: RGB, HSV, YCbCr, LAB
        """
        conversions = {
            'HSV': cv2.COLOR_BGR2HSV,
            'YCbCr': cv2.COLOR_BGR2YCrCb,
            'LAB': cv2.COLOR_BGR2LAB,
            'RGB': cv2.COLOR_BGR2RGB
        }
        
        if target_model in conversions:
            converted = cv2.cvtColor(image, conversions[target_model])
            return converted
        else:
            return image
    
    def viola_jones_face_detection(self, image):
        """
        Viola-Jones face detection algorithm
        Returns face coordinates and enhanced face regions
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        result_image = image.copy()
        face_regions = []
        
        # Check if cascades are available
        if self.face_cascade is None:
            print("⚠️ Face detection unavailable - cascade not loaded")
            cv2.putText(result_image, 'Face detection unavailable', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return result_image, face_regions
            
        try:
            # Detect faces using Haar Cascade (Viola-Jones method)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            # Detect eyes for additional facial feature analysis
            eyes = []
            if self.eye_cascade is not None:
                eyes = self.eye_cascade.detectMultiScale(gray)
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Extract face region for emotion analysis
                face_roi = image[y:y+h, x:x+w]
                face_regions.append({
                    'coordinates': (x, y, w, h),
                    'roi': face_roi,
                    'confidence': self._calculate_face_confidence(face_roi)
                })
                
                # Label the face
                cv2.putText(result_image, 'Face', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw eye detections
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(result_image, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
                
        except Exception as e:
            print(f"⚠️ Error during face detection: {e}")
            cv2.putText(result_image, 'Face detection error', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result_image, face_regions
    
    def _calculate_face_confidence(self, face_roi):
        """Calculate confidence score for detected face region"""
        if face_roi.size == 0:
            return 0.0
            
        # Simple confidence based on face size and contrast
        height, width = face_roi.shape[:2]
        size_score = min(1.0, (height * width) / (100 * 100))
        
        if len(face_roi.shape) == 3:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_roi
            
        contrast_score = gray_face.std() / 128.0  # Normalize contrast
        
        return min(1.0, (size_score + contrast_score) / 2.0)
    
    def enhance_emotion_features(self, image):
        """
        Apply multiple image processing techniques to enhance emotion detection
        Combines edge detection, color enhancement, and facial feature analysis
        """
        try:
            # Convert to different color models for analysis
            hsv = self.color_model_conversion(image, 'HSV')
            ycbcr = self.color_model_conversion(image, 'YCbCr')
            
            # Edge detection for facial structure enhancement
            edges = self.edge_detection(image, 'canny')
            
            # Face detection using Viola-Jones
            face_detected, face_regions = self.viola_jones_face_detection(image)
            
            # Enhance contrast in luminance channel (Y in YCbCr)
            enhanced_y = cv2.equalizeHist(ycbcr[:,:,0])
            ycbcr_enhanced = ycbcr.copy()
            ycbcr_enhanced[:,:,0] = enhanced_y
            enhanced_image = cv2.cvtColor(ycbcr_enhanced, cv2.COLOR_YCrCb2BGR)
            
            cascade_status = "available" if self.face_cascade is not None else "unavailable"
            
            return {
                'original': image,
                'enhanced': enhanced_image,
                'edges': edges,
                'hsv': hsv,
                'ycbcr': ycbcr,
                'face_detected': face_detected,
                'face_regions': face_regions,
                'processing_info': {
                    'faces_detected': len(face_regions),
                    'enhancement_applied': True,
                    'color_models': ['HSV', 'YCbCr'],
                    'edge_detection': 'Canny',
                    'face_cascade_status': cascade_status
                }
            }
        except Exception as e:
            print(f"⚠️ Error in emotion feature enhancement: {e}")
            # Return minimal processing result
            return {
                'original': image,
                'enhanced': image.copy(),
                'edges': np.zeros(image.shape[:2], dtype=np.uint8),
                'hsv': image.copy(),
                'ycbcr': image.copy(),
                'face_detected': image.copy(),
                'face_regions': [],
                'processing_info': {
                    'faces_detected': 0,
                    'enhancement_applied': False,
                    'error': str(e)
                }
            }
    
    def histogram_analysis(self, image):
        """
        Analyze image histogram for emotion-relevant features
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Calculate histogram features
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        brightness = mean_intensity / 255.0
        contrast = std_intensity / 128.0
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        return {
            'histogram': hist,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'brightness': brightness,
            'contrast': contrast,
            'equalized': equalized
        }
    
    def spatial_filtering(self, image, filter_type='gaussian'):
        """
        Apply spatial filters for image enhancement
        """
        if filter_type == 'gaussian':
            filtered = cv2.GaussianBlur(image, (5, 5), 1.0)
        elif filter_type == 'median':
            filtered = cv2.medianBlur(image, 5)
        elif filter_type == 'bilateral':
            filtered = cv2.bilateralFilter(image, 9, 75, 75)
        else:
            filtered = image.copy()
            
        return filtered

def encode_image_to_base64(image):
    """Convert OpenCV image to base64 for web display"""
    _, buffer = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def decode_base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_data = base64.b64decode(base64_string)
    img_pil = Image.open(BytesIO(img_data))
    img_array = np.array(img_pil)
    
    if len(img_array.shape) == 3:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_cv = img_array
        
    return img_cv