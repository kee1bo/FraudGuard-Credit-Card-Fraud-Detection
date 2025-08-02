#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add src to Python path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

try:
    from app.main import create_app
    
    app = create_app()
    print(" Starting FraudGuard AI...")
    print(" Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all project files are in place")
except Exception as e:
    print(f"❌ Error: {e}")