"""
Y.M.I.R Gemini API Key Rotation Manager
Automatically cycles through multiple Gemini API keys when quotas are exhausted
"""

import os
import json
import time
import random
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import google.generativeai as genai


@dataclass
class APIKeyStatus:
    """Track API key usage and status"""
    key_id: str
    key: str
    is_active: bool = True
    quota_exhausted: bool = False
    last_error: Optional[str] = None
    error_count: int = 0
    last_used: Optional[datetime] = None
    reset_time: Optional[datetime] = None


class GeminiAPIManager:
    """
    Manages multiple Gemini API keys with automatic rotation
    Features:
    - Automatic failover when quota exhausted
    - Error tracking and recovery
    - Smart rotation to distribute load
    - Persistent state management
    """
    
    def __init__(self, state_file: str = "gemini_api_state.json"):
        self.state_file = state_file
        self.api_keys: List[APIKeyStatus] = []
        self.current_key_index = 0
        self.load_api_keys()
        self.load_state()
        # Reset quotas on startup to give fresh start
        self.reset_all_quotas()
        
    def load_api_keys(self):
        """Load all Gemini API keys from environment"""
        api_keys = []
        
        # Load numbered keys (GEMINI_API_KEY1, GEMINI_API_KEY2, etc.)
        for i in range(1, 9):  # Keys 1-8
            key = os.getenv(f'GEMINI_API_KEY{i}')
            if key:
                api_keys.append(APIKeyStatus(
                    key_id=f"GEMINI_API_KEY{i}",
                    key=key
                ))
                print(f"🔑 Loaded {f'GEMINI_API_KEY{i}'}: {key[:20]}...")
        
        # Load primary key as fallback
        primary_key = os.getenv('GEMINI_API_KEY')
        if primary_key and primary_key not in [k.key for k in api_keys]:
            api_keys.append(APIKeyStatus(
                key_id="GEMINI_API_KEY",
                key=primary_key
            ))
        
        if not api_keys:
            raise ValueError("❌ No Gemini API keys found in environment variables")
        
        self.api_keys = api_keys
        print(f"✅ Loaded {len(self.api_keys)} Gemini API keys")
        
    def load_state(self):
        """Load persistent state from file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    
                self.current_key_index = state.get('current_key_index', 0)
                
                # Update key statuses
                for key_data in state.get('api_keys', []):
                    for api_key in self.api_keys:
                        if api_key.key_id == key_data['key_id']:
                            api_key.is_active = key_data.get('is_active', True)
                            api_key.quota_exhausted = key_data.get('quota_exhausted', False)
                            api_key.error_count = key_data.get('error_count', 0)
                            api_key.last_error = key_data.get('last_error')
                            if key_data.get('last_used'):
                                api_key.last_used = datetime.fromisoformat(key_data['last_used'])
                            if key_data.get('reset_time'):
                                api_key.reset_time = datetime.fromisoformat(key_data['reset_time'])
                            break
                            
                print("📄 Loaded API key state from file")
        except Exception as e:
            print(f"⚠️ Could not load state file: {e}")
            
    def save_state(self):
        """Save current state to file"""
        try:
            state = {
                'current_key_index': self.current_key_index,
                'last_updated': datetime.now().isoformat(),
                'api_keys': []
            }
            
            for api_key in self.api_keys:
                key_data = {
                    'key_id': api_key.key_id,
                    'is_active': api_key.is_active,
                    'quota_exhausted': api_key.quota_exhausted,
                    'error_count': api_key.error_count,
                    'last_error': api_key.last_error,
                    'last_used': api_key.last_used.isoformat() if api_key.last_used else None,
                    'reset_time': api_key.reset_time.isoformat() if api_key.reset_time else None
                }
                state['api_keys'].append(key_data)
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Could not save state: {e}")
    
    def get_current_api_key(self) -> Optional[APIKeyStatus]:
        """Get the current active API key"""
        if not self.api_keys:
            return None
            
        # Check if current key is valid
        current_key = self.api_keys[self.current_key_index]
        
        # Reset quota if enough time has passed (24 hours)
        if (current_key.quota_exhausted and current_key.reset_time and 
            datetime.now() > current_key.reset_time):
            current_key.quota_exhausted = False
            current_key.error_count = 0
            current_key.last_error = None
            print(f"🔄 Reset quota for {current_key.key_id}")
            
        # If current key is exhausted, rotate to next
        if current_key.quota_exhausted or not current_key.is_active:
            return self.rotate_to_next_key()
            
        return current_key
    
    def rotate_to_next_key(self) -> Optional[APIKeyStatus]:
        """Rotate to the next available API key"""
        original_index = self.current_key_index
        attempts = 0
        
        while attempts < len(self.api_keys):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            current_key = self.api_keys[self.current_key_index]
            
            # Reset quota if enough time has passed
            if (current_key.quota_exhausted and current_key.reset_time and 
                datetime.now() > current_key.reset_time):
                current_key.quota_exhausted = False
                current_key.error_count = 0
                current_key.last_error = None
                print(f"🔄 Reset quota for {current_key.key_id}")
            
            # Check if this key is usable
            if current_key.is_active and not current_key.quota_exhausted:
                print(f"🔄 Rotated to API key: {current_key.key_id}")
                self.save_state()
                return current_key
                
            attempts += 1
        
        # All keys exhausted
        print("❌ All Gemini API keys are exhausted!")
        return None
    
    def mark_key_exhausted(self, error_message: str = None):
        """Mark current key as quota exhausted"""
        current_key = self.api_keys[self.current_key_index]
        current_key.quota_exhausted = True
        current_key.error_count += 1
        current_key.last_error = error_message
        current_key.reset_time = datetime.now() + timedelta(hours=24)
        
        print(f"❌ Marked {current_key.key_id} as exhausted: {error_message}")
        self.save_state()
    
    def mark_key_error(self, error_message: str):
        """Mark current key as having an error"""
        current_key = self.api_keys[self.current_key_index]
        current_key.error_count += 1
        current_key.last_error = error_message
        
        # If too many errors, temporarily disable
        if current_key.error_count >= 5:
            current_key.is_active = False
            current_key.reset_time = datetime.now() + timedelta(hours=1)
            print(f"🔥 Temporarily disabled {current_key.key_id} due to errors")
        
        self.save_state()
    
    def configure_genai(self) -> bool:
        """Configure Google GenAI with current API key"""
        try:
            api_key_status = self.get_current_api_key()
            if not api_key_status:
                print("❌ No available Gemini API keys")
                # Print status of all keys for debugging
                for key in self.api_keys:
                    print(f"   {key.key_id}: active={key.is_active}, exhausted={key.quota_exhausted}")
                return False
            
            # Configure the API
            genai.configure(api_key=api_key_status.key)
            api_key_status.last_used = datetime.now()
            
            print(f"✅ Configured Gemini with {api_key_status.key_id} (key #{self.current_key_index + 1})")
            self.save_state()
            return True
            
        except Exception as e:
            print(f"❌ Failed to configure Gemini: {e}")
            return False
    
    def create_model(self, model_name: str = "gemini-2.0-flash-lite", **kwargs):
        """Create a Gemini model with the current API key"""
        if not self.configure_genai():
            raise Exception("No available Gemini API keys")
        
        try:
            model = genai.GenerativeModel(model_name, **kwargs)
            print(f"✅ Created Gemini model: {model_name}")
            return model
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for quota exhaustion with more patterns
            if ('quota' in error_msg or 'limit' in error_msg or 'exhausted' in error_msg or 
                '429' in error_msg or 'exceeded' in error_msg):
                print(f"🔥 Quota exhausted for {self.api_keys[self.current_key_index].key_id}")
                self.mark_key_exhausted(str(e))
                # Try with next key
                next_key = self.rotate_to_next_key()
                if next_key:
                    print(f"🔄 Retrying with {next_key.key_id}")
                    return self.create_model(model_name, **kwargs)
                else:
                    print("❌ All API keys exhausted!")
                    raise Exception("All Gemini API keys are exhausted")
            else:
                self.mark_key_error(str(e))
            
            raise e
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all API keys"""
        status = {
            'current_key': self.api_keys[self.current_key_index].key_id,
            'total_keys': len(self.api_keys),
            'active_keys': len([k for k in self.api_keys if k.is_active and not k.quota_exhausted]),
            'exhausted_keys': len([k for k in self.api_keys if k.quota_exhausted]),
            'keys': []
        }
        
        for key in self.api_keys:
            key_info = {
                'key_id': key.key_id,
                'is_active': key.is_active,
                'quota_exhausted': key.quota_exhausted,
                'error_count': key.error_count,
                'last_error': key.last_error,
                'last_used': key.last_used.isoformat() if key.last_used else None,
                'reset_time': key.reset_time.isoformat() if key.reset_time else None
            }
            status['keys'].append(key_info)
        
        return status
    
    def reset_all_quotas(self):
        """Reset all quota exhaustions (emergency recovery)"""
        reset_count = 0
        for key_status in self.api_keys:
            if key_status.quota_exhausted:
                key_status.quota_exhausted = False
                key_status.error_count = 0
                key_status.last_error = None
                key_status.reset_time = None
                reset_count += 1
                print(f"🔄 Emergency reset for {key_status.key_id}")
        
        if reset_count > 0:
            print(f"🚨 Emergency reset {reset_count} API keys")
            self.save_state()
        return reset_count


# Global instance
gemini_manager = GeminiAPIManager()


def get_gemini_model(model_name: str = "gemini-2.0-flash-lite", **kwargs):
    """
    Get a Gemini model with automatic API key rotation
    Usage: model = get_gemini_model("gemini-2.0-flash-lite")
    """
    return gemini_manager.create_model(model_name, **kwargs)


def get_api_status():
    """Get current API key status"""
    return gemini_manager.get_status()


def reset_all_api_quotas():
    """Reset all quota exhaustions across all keys"""
    return gemini_manager.reset_all_quotas()


if __name__ == "__main__":
    # Test the API manager
    print("🧪 Testing Gemini API Manager...")
    
    try:
        model = get_gemini_model()
        print("✅ Successfully created Gemini model")
        
        # Print status
        status = get_api_status()
        print(f"📊 Status: {status['active_keys']}/{status['total_keys']} keys active")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")