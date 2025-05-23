#!/usr/bin/env python3
"""
Text Redaction Script

This script detects inappropriate words in images and redacts them with black rectangles.
Uses OCR (Optical Character Recognition) to detect text and filters profanity.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw
import argparse
import re
from typing import List, Tuple, Dict

class TextRedactor:
    def __init__(self):
        # List of inappropriate words to redact
        self.inappropriate_words = {
            'fuck', 'fucking', 'shit', 'damn', 'hell', 'ass', 'bitch', 'bastard',
            'crap', 'piss', 'cock', 'dick', 'pussy', 'whore', 'slut', 'fag',
            'nigger', 'nigga', 'retard', 'asshole', 'motherfucker', 'bullshit'
        }
    
    def add_words(self, words: List[str]):
        """Add additional words to the inappropriate words list."""
        self.inappropriate_words.update(word.lower() for word in words)
    
    def detect_text_with_coordinates(self, image_path: str) -> List[Dict]:
        """
        Detect text in image and return coordinates along with text content.
        
        Returns:
            List of dictionaries containing text, coordinates, and bounding boxes
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB for pytesseract
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use pytesseract to get detailed text information
        data = pytesseract.image_to_data(rgb_image, output_type=pytesseract.Output.DICT)
        
        text_blocks = []
        n_boxes = len(data['level'])
        
        for i in range(n_boxes):
            # Filter out empty text and low confidence detections
            if int(data['conf'][i]) > 30 and data['text'][i].strip():
                text_block = {
                    'text': data['text'][i],
                    'left': data['left'][i],
                    'top': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'confidence': data['conf'][i]
                }
                text_blocks.append(text_block)
        
        return text_blocks
    
    def contains_inappropriate_content(self, text: str) -> bool:
        """Check if text contains inappropriate words."""
        # Clean text and convert to lowercase
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = clean_text.split()
        
        # Check each word
        for word in words:
            if word in self.inappropriate_words:
                return True
            
            # Check for partial matches (words containing inappropriate content)
            for inappropriate_word in self.inappropriate_words:
                if inappropriate_word in word:
                    return True
        
        return False
    
    def redact_image(self, image_path: str, output_path: str = None) -> str:
        """
        Redact inappropriate words in image with black rectangles.
        
        Args:
            image_path: Path to input image
            output_path: Path for output image (optional)
            
        Returns:
            Path to the redacted image
        """
        if output_path is None:
            output_path = image_path.replace('.', '_redacted.')
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Get text blocks with coordinates
        text_blocks = self.detect_text_with_coordinates(image_path)
        
        redacted_count = 0
        
        # Process each text block
        for block in text_blocks:
            if self.contains_inappropriate_content(block['text']):
                # Draw black rectangle over inappropriate text
                x = block['left']
                y = block['top']
                w = block['width']
                h = block['height']
                
                # Add some padding to ensure full coverage
                padding = 5
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = w + (2 * padding)
                h = h + (2 * padding)
                
                # Draw black rectangle
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
                redacted_count += 1
                
                print(f"Redacted: '{block['text']}' at ({x}, {y}, {w}, {h})")
        
        # Save redacted image
        cv2.imwrite(output_path, image)
        
        print(f"Redaction complete. {redacted_count} inappropriate text blocks redacted.")
        print(f"Output saved to: {output_path}")
        
        return output_path
    
    def preview_detections(self, image_path: str, output_path: str = None) -> str:
        """
        Create a preview image showing all detected text with bounding boxes.
        Inappropriate content is highlighted in red, normal content in green.
        """
        if output_path is None:
            output_path = image_path.replace('.', '_preview.')
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Get text blocks
        text_blocks = self.detect_text_with_coordinates(image_path)
        
        # Draw bounding boxes
        for block in text_blocks:
            x = block['left']
            y = block['top']
            w = block['width']
            h = block['height']
            
            # Choose color based on content
            if self.contains_inappropriate_content(block['text']):
                color = (0, 0, 255)  # Red for inappropriate
                thickness = 3
            else:
                color = (0, 255, 0)  # Green for appropriate
                thickness = 2
            
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            
            # Add text label
            cv2.putText(image, f"'{block['text']}'", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save preview
        cv2.imwrite(output_path, image)
        print(f"Preview saved to: {output_path}")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(description='Redact inappropriate text in images')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('-o', '--output', help='Output path for redacted image')
    parser.add_argument('-p', '--preview', action='store_true', 
                       help='Generate preview showing detected text boundaries')
    parser.add_argument('--add-words', nargs='+', 
                       help='Additional inappropriate words to filter')
    
    args = parser.parse_args()
    
    # Create redactor
    redactor = TextRedactor()
    
    # Add custom words if provided
    if args.add_words:
        redactor.add_words(args.add_words)
        print(f"Added {len(args.add_words)} custom words to filter list")
    
    try:
        if args.preview:
            # Generate preview
            preview_path = redactor.preview_detections(args.image_path, args.output)
            print(f"Preview generated: {preview_path}")
        else:
            # Perform redaction
            output_path = redactor.redact_image(args.image_path, args.output)
            print(f"Redaction completed: {output_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())