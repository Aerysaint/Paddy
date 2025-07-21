#!/usr/bin/env python3

import sys
sys.path.insert(0, 'src')

try:
    from data_models import HeadingCandidate, TextBlock
    print("✓ data_models import successful")
except ImportError as e:
    print(f"✗ data_models import failed: {e}")

try:
    from logging_config import setup_logging
    print("✓ logging_config import successful")
except ImportError as e:
    print(f"✗ logging_config import failed: {e}")

try:
    from heading_level_classifier import HeadingLevelClassifier
    print("✓ HeadingLevelClassifier import successful")
    
    # Test instantiation
    classifier = HeadingLevelClassifier()
    print("✓ HeadingLevelClassifier instantiation successful")
    
except ImportError as e:
    print(f"✗ HeadingLevelClassifier import failed: {e}")
except Exception as e:
    print(f"✗ HeadingLevelClassifier instantiation failed: {e}")