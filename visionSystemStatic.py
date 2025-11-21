"""
1. Loads ONE image from a file
2. Runs YOLO11 AI to detect objects
3. Shows the result with boxes
4. Saves the result as a new image
"""

# Import the libraries we need
import cv2                    # For handling images
from ultralytics.models import YOLO# For the YOLO11 AI model

def main():

    
    print("="*60)
    print("STATIC IMAGE TEST - YOLO11")
    print("="*60)
    print("This will test the AI model on a single image\n")
    
    print("[Step 1/4] Loading AI Model...")
    print("  Model: YOLO11 nano")
    
    try:
        model = YOLO('yolo11n.pt')
        print("Model loaded successfully!\n")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have internet for first-time download")
        return
    
    
    print("[Step 2/4] Loading test image...")
    
    # Image filename to test
    # This can change this to any .jpg or .png file you have
    image_path = 'test_image.jpg'
    
    # Read the image file
    # This loads the image into computer memory as a matrix of numbers
    frame = cv2.imread(image_path)
    
    # Check if the image loaded successfully
    if frame is None:
        print(f"Error: Could not load '{image_path}'")
        print("\nTo fix this:")
        print("1. Make sure 'test_image.jpg' exists in this folder")
        print("2. Or download a sample image:")
        print("wget https://ultralytics.com/images/bus.jpg -O test_image.jpg")
        print("3. Or change 'image_path' in the code to your image file")
        return
    
    # Get image dimensions
    height = frame.shape[0]  # Number of rows (pixels tall)
    width = frame.shape[1]   # Number of columns (pixels wide)
    
    print(f"Image loaded successfully")
    print(f"File: {image_path}")
    print(f"Size: {width} x {height} pixels\n")
    
    
    print("[Step 3/4] Running object detection...")
    print("  (This might take a few seconds...)")
    
    # Run the AI model on the image
    results = model(frame, verbose=False)
    
    # Draw boxes around detected objects
    annotated_frame = results[0].plot()
    
    print("Detection complete!\n")
    
    print("[Step 4/4] Showing results...")
    
    # Display the image with detected objects
    print("A window will open showing the results")
    print("Press ANY KEY to close the window\n")
    
    cv2.imshow("YOLO11 Detection Results", annotated_frame)
    cv2.waitKey(0)  # Wait forever until user presses a key
    cv2.destroyAllWindows()
    
    # Save the result to a new file
    output_path = 'detection_result.jpg'
    cv2.imwrite(output_path, annotated_frame)
    print(f"  ✓ Result saved to: {output_path}\n")
    

    print("="*60)
    print("DETECTED OBJECTS:")
    print("="*60)
    
    # Get the list of detected objects from the results
    detections = results[0].boxes
    
    if len(detections) == 0:
        print("  No objects detected in this image")
        print("  (Try an image with people, cars, or common objects)")
    else:
        # Loop through each detected object and print info
        for i, detection in enumerate(detections):
            # Get the class ID (which type of object: 0=person, 2=car, etc.)
            class_id = int(detection.cls[0])
            
            # Get confidence score (how sure the AI is)
            confidence = float(detection.conf[0])
            
            # Get the class name (convert ID to readable name like "person")
            class_name = results[0].names[class_id]
            
            # Print the detection
            print(f"  {i+1}. {class_name}")
            print(f"     Confidence: {confidence*100:.1f}%")
    
    print("="*60)
    print("\nTest complete! If this worked, your setup is ready.")
    print("Next step: Try 'vision_system.py' for live camera detection")
    print("="*60)


if __name__ == "__main__":
    main()