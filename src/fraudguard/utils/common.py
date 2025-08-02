import os
import sys
import pickle
import joblib
import yaml
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

def save_object(obj: Any, file_path: str) -> None:
    """Save an object to a file using joblib"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)
            
    except Exception as e:
        raise Exception(f"Error saving object to {file_path}: {str(e)}")

def load_object(file_path: str) -> Any:
    """Load an object from a file using joblib"""
    try:
        with open(file_path, "rb") as file_obj:
            return joblib.load(file_obj)
            
    except Exception as e:
        raise Exception(f"Error loading object from {file_path}: {str(e)}")

def save_json(data: Dict, file_path: str) -> None:
    """Save data as JSON file"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4, default=str)
            
    except Exception as e:
        raise Exception(f"Error saving JSON to {file_path}: {str(e)}")

def load_json(file_path: str) -> Dict:
    """Load data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
            
    except Exception as e:
        raise Exception(f"Error loading JSON from {file_path}: {str(e)}")

def load_yaml(file_path: str) -> Dict:
    """Load configuration from YAML file"""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
            
    except Exception as e:
        raise Exception(f"Error loading YAML from {file_path}: {str(e)}")

def create_directories(path_list: List[str]) -> None:
    """Create multiple directories"""
    for path in path_list:
        os.makedirs(path, exist_ok=True)
