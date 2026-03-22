import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
import argparse
import os

# Set tesseract path - change this to match your installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class UseCaseDiagramDetector:
    def is_oval(self, contour, image_shape):
        # Need at least 5 points to fit ellipse
        if len(contour) < 5:
            return False, None
        
        try:
            area = cv2.contourArea(contour)
            # Skip tiny contours
            if area < 1000:
                return False, None
                
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Fit ellipse
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (ew, eh), angle = ellipse
            
            # Calculate ellipse area
            ellipse_area = np.pi * (ew/2) * (eh/2)
            area_ratio = area / ellipse_area if ellipse_area > 0 else 0
            
            # Use case ovals are roughly elliptical but not perfect circles
            is_valid = (
                0.7 < area_ratio < 1.3 and      # Reasonably close to ellipse shape
                0.7 < aspect_ratio < 3.0 and     # Not too round, not too thin
                0.4 < circularity < 0.95 and     # Not too irregular
                area > 500 and area < 20000 and  # Size constraints
                len(contour) > 8                 # Enough points
            )
            
            if is_valid:
                return True, {
                    "ellipse": ellipse,
                    "area": area,
                    "perimeter": perimeter,
                    "circularity": circularity,
                    "aspect_ratio": aspect_ratio,
                    "area_ratio": area_ratio,
                    "bounding_rect": (x, y, w, h)
                }
            return False, None
                
        except:
            # Silently fail if we can't fit ellipse
            return False, None
    
    def detect_elements(self, image_path):
        """Detect actors, use cases and system boundaries in a UML diagram"""
        # Load and resize image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
            
        # Make image a consistent size for processing
        img = cv2.resize(img, (800, 750), interpolation=cv2.INTER_AREA)
        annotated = img.copy()
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try two thresholding approaches and combine them
        _, binary1 = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        binary = cv2.bitwise_or(binary1, binary2)
        
        # Clean up noise
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Get text with OCR
        text_data = pytesseract.image_to_data(gray, output_type=Output.DICT)
        
        # Find contours - these will be our diagram elements
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug image
        debug_dir = os.path.dirname(image_path) or '.'
        cv2.imwrite(os.path.join(debug_dir, 'binary_preprocessed.jpg'), binary)
        
        # Lists to store detected elements
        use_cases = []
        actors = []
        system_boundaries = []
        
        # Process each contour
        for cnt in contours:
            # Skip tiny contours - likely noise
            if cv2.contourArea(cnt) < 100:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Calculate shape metrics
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Simplify contour
            epsilon = 0.03 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Check for use case oval
            is_oval, metrics = self.is_oval(cnt, img.shape)
            
            if is_oval:
                # Found a use case (oval)
                x, y, w, h = metrics["bounding_rect"]
                use_cases.append((x, y, w, h, cnt, metrics))
                
                # Draw oval on output image
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                try:
                    cv2.ellipse(annotated, metrics["ellipse"], (0, 255, 0), 1)
                except:
                    # Fallback if ellipse drawing fails
                    cv2.drawContours(annotated, [cnt], 0, (0, 255, 0), 1)
                
                cv2.putText(annotated, "Use Case", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Check for system boundary - large rectangle
            elif len(approx) == 4 and w > 200 and h > 200:
                system_boundaries.append((x, y, w, h, cnt))
                cv2.drawContours(annotated, [cnt], 0, (255, 0, 0), 2)
                cv2.putText(annotated, "System Boundary", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Check for actor - tall thin shape
            elif 1.5 < h/w < 4 and circularity < 0.5:
                actors.append((x, y, w, h, cnt))
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.drawContours(annotated, [cnt], 0, (0, 0, 255), 1)
                cv2.putText(annotated, "Actor", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Highlight text labels
        for i in range(len(text_data['text'])):
            text = text_data['text'][i].strip()
            if text:
                x, y = text_data['left'][i], text_data['top'][i]
                w, h = text_data['width'][i], text_data['height'][i]
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 1)
                cv2.putText(annotated, f"Label: {text}", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return {
            'original_image': img,
            'annotated_image': annotated,
            'use_cases': use_cases,
            'actors': actors,
            'system_boundaries': system_boundaries
        }
    
    def display_results(self, results):
        """Show results in a matplotlib window"""
        try:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(results['original_image'], cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(results['annotated_image'], cv2.COLOR_BGR2RGB))
            plt.title('Detected Elements')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display results: {e}")
        
    def save_results(self, results, output_path):
        """Save annotated image to disk"""
        cv2.imwrite(output_path, results['annotated_image'])


def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Detect elements in UML use case diagrams')
    
    # Add arguments
    parser.add_argument('--image', '-i', type=str, required=False, default='case1.png',
                      help='Path to the input image file')
    parser.add_argument('--output', '-o', type=str, required=False, default='annotated_diagram.jpg',
                      help='Path to save the annotated output image')
    parser.add_argument('--no-display', action='store_true',
                      help='Do not display the result window')
    parser.add_argument('--tesseract', '-t', type=str,
                      help='Path to tesseract executable')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set custom tesseract path if provided
    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract
    
    # Check if input file exists
    if not os.path.exists(args.image):
        print(f"Error: Input file '{args.image}' not found")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    detector = UseCaseDiagramDetector()
    
    try:
        # Find elements in the diagram
        results = detector.detect_elements(args.image)
        
        # Show results (unless disabled)
        if not args.no_display:
            detector.display_results(results)
        
        # Save annotated image
        detector.save_results(results, args.output)
        print(f"Saved annotated image to: {args.output}")
        
        # Print stats
        print(f"Found {len(results['use_cases'])} use cases")
        print(f"Found {len(results['actors'])} actors")
        print(f"Found {len(results['system_boundaries'])} system boundaries")
        
    except Exception as e:
        print(f"Error processing diagram: {e}")


if __name__ == "__main__":
    main()