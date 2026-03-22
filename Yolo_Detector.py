import numpy as np
import cv2
from ultralytics import YOLO
import os

# Load YOLO model 
model = YOLO("runs/detect/train6/weights/best.pt")

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Can't open {image_path}")
    return img

def process_diagram(image_path):
    # Load image
    img = load_image(image_path)
    output_img = img.copy()
    
    # Create debug dir
    debug_dir = os.path.join(os.path.dirname(image_path), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Initialize element lists
    use_cases = []
    actors = []
    bounding_boxes = []

    # Run YOLO detection
    results = model(img, conf=0.3)
    print(f"Found {len(results[0].boxes)} objects with YOLO")
    
    # Process YOLO results
    for result in results:
        boxes = []
        
        for box in result.boxes:
            if len(box.xyxy) == 0:
                continue
                
            # Get box info
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # Skip if not an actor
            if class_id != 0:
                continue
                
            # Basic sanity checks on box
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0
            
            # Skip weird shaped boxes
            if not (0.5 < aspect_ratio < 2.0):
                continue
                
            # Skip huge boxes
            if width > img.shape[1]//2 or height > img.shape[0]//2:
                continue
            
            # Check if duplicate
            is_dupe = False
            for ex1, ey1, ex2, ey2 in boxes:
                # Calculate overlap
                x_left = max(x1, ex1)
                y_top = max(y1, ey1)
                x_right = min(x2, ex2)
                y_bottom = min(y2, ey2)
                
                if x_right > x_left and y_bottom > y_top:
                    # Calculate IoU
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (ex2 - ex1) * (ey2 - ey1)
                    iou = intersection / float(area1 + area2 - intersection)
                    
                    if iou > 0.3:
                        is_dupe = True
                        break
            
            if not is_dupe:
                boxes.append((x1, y1, x2, y2))
                actors.append((x1, y1, width, height))
                cv2.rectangle(output_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(output_img, f"Actor {confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
    canny = cv2.Canny(gray, 50, 150)
    
    # Save debug images
    cv2.imwrite(os.path.join(debug_dir, "debug_adaptive.jpg"), adaptive)
    cv2.imwrite(os.path.join(debug_dir, "debug_canny.jpg"), canny)
    
    # Find contours from edges
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Debug contours
    contour_viz = img.copy()
    cv2.drawContours(contour_viz, contours, -1, (0, 255, 255), 2)
    cv2.imwrite(os.path.join(debug_dir, "all_contours.jpg"), contour_viz)
    
    # Process contours
    for c in contours:
        # Skip small contours
        if cv2.contourArea(c) < 1000:
            continue
            
        # Get simplified contour and bounding box
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        
        # Special case for system boundary
        if len(approx) == 8 and area > 25000 and w > 150 and h > 150:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if not (0.4 <= circularity <= 0.8):
                bounding_boxes.append((x, y, w, h))
                cv2.putText(output_img, "System", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.drawContours(output_img, [c], 0, (0, 0, 255), 2)
                continue
        
        # Check for ovals (use cases)
        if 6 <= len(approx) <= 20:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            if 0.4 <= circularity <= 0.8:
                use_cases.append((x, y, w, h))
                cv2.putText(output_img, "Use Case", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.drawContours(output_img, [c], 0, (0, 255, 0), 2)
                continue
        
        # Check for rectangles (system boundaries)
        if 4 <= len(approx) <= 10:
            rectangularity = area / (w * h) if w*h > 0 else 0
            
            if 0.6 <= rectangularity <= 1.4 and area > 10000:  
                bounding_boxes.append((x, y, w, h))
                cv2.putText(output_img, "System", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.drawContours(output_img, [c], 0, (0, 0, 255), 2)
    
    # Fallback actor detection
    if len(actors) == 0:
        for c in contours:
            area = cv2.contourArea(c)
            if 500 < area < 5000:  
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h if h > 0 else 0
                if 0.3 < aspect_ratio < 0.8 and h > w:  
                    actors.append((x, y, w, h))
                    cv2.rectangle(output_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(output_img, "Actor", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    break
    
    # Save result
    out_path = os.path.join(os.path.dirname(image_path), 
                           "processed_" + os.path.basename(image_path))
    cv2.imwrite(out_path, output_img)
    print(f"Output saved to: {out_path}")
    
    # Print stats
    print(f"Found: {len(use_cases)} use cases, {len(actors)} actors, {len(bounding_boxes)} systems")
    
    return output_img, use_cases, actors, bounding_boxes

def detect_relationships(image_path, use_cases, actors, bounds):
    img = load_image(image_path)
    output = img.copy()
    
    # Edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Find lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                           threshold=50, minLineLength=50, maxLineGap=10)
    
    relations = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Draw line
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
            relations.append(((x1, y1), (x2, y2)))
    
    # Save output
    out_path = os.path.join(os.path.dirname(image_path), 
                           "relations_" + os.path.basename(image_path))
    cv2.imwrite(out_path, output)
    
    return relations

if __name__ == "__main__":
    try:
        # Update this to your image path
        img_path = 'case5.png'
        
        # Process diagram
        result, use_cases, actors, bounds = process_diagram(img_path)
        
        # Find relationships
        relations = detect_relationships(img_path, use_cases, actors, bounds)
        
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()