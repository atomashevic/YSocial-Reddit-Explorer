#!/usr/bin/env python3
"""
Test script for the text redactor
"""

from text_redactor import TextRedactor
import sys
import os

def test_redactor():
    """Test the text redactor with a sample image."""
    
    # Check if we have an image to test with
    test_image = None
    for ext in ['.png', '.jpg', '.jpeg']:
        if os.path.exists(f'sample{ext}'):
            test_image = f'sample{ext}'
            break
    
    if not test_image:
        print("No test image found. Please save an image as 'sample.png', 'sample.jpg', or 'sample.jpeg'")
        return False
    
    print(f"Testing with image: {test_image}")
    
    try:
        # Create redactor
        redactor = TextRedactor()
        
        # Generate preview first
        print("Generating preview...")
        preview_path = redactor.preview_detections(test_image)
        
        # Perform redaction
        print("Performing redaction...")
        output_path = redactor.redact_image(test_image)
        
        print(f"Success! Check these files:")
        print(f"  Preview: {preview_path}")
        print(f"  Redacted: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_redactor()
    sys.exit(0 if success else 1)