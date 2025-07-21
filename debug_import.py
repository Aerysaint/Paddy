#!/usr/bin/env python3

import sys
import traceback
sys.path.insert(0, 'src')

print("Attempting to import heading_level_classifier module...")

try:
    import heading_level_classifier
    print("Module imported successfully")
    print("Module contents:", dir(heading_level_classifier))
    
    # Try to access the class
    if hasattr(heading_level_classifier, 'HeadingLevelClassifier'):
        print("HeadingLevelClassifier found in module")
        cls = heading_level_classifier.HeadingLevelClassifier
        print("Class:", cls)
    else:
        print("HeadingLevelClassifier NOT found in module")
        
except Exception as e:
    print("Error importing module:", e)
    traceback.print_exc()

print("\nTrying to execute file directly...")
try:
    with open('src/heading_level_classifier.py', 'r') as f:
        code = f.read()
    
    # Create a namespace to execute in
    namespace = {}
    exec(code, namespace)
    
    print("File executed successfully")
    print("Namespace contents:", [k for k in namespace.keys() if not k.startswith('__')])
    
    if 'HeadingLevelClassifier' in namespace:
        print("HeadingLevelClassifier found in namespace")
        cls = namespace['HeadingLevelClassifier']
        print("Class:", cls)
        instance = cls()
        print("Instance created:", instance)
    else:
        print("HeadingLevelClassifier NOT found in namespace")
        
except Exception as e:
    print("Error executing file:", e)
    traceback.print_exc()