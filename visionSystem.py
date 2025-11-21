"""
1. Opens your webcam
2. Uses YOLO11 AI to detect objects in real-time
3. Draws boxes around detected objects
4. Shows the video with detections on screen

Hardware needed:
- Raspberry Pi (any model with USB ports)
- USB Webcam    
"""

# Import libraries (pre-installed code packages we need)
import cv2                   
from ultralytics.models import YOLO 
import time                   # not used yet


def main():

    print("="*60)
    print("VISION SYSTEM - STARTING UP")
    print("="*60)
    print("This system will:")
    print("  1. Open your camera")
    print("  2. Detect objects using YOLO11 AI")
    print("  3. Show results in real-time")
    print("="*60 + "\n")
    
    
    print("[Step 1/3] Loading AI Model...")
    print("  Model: YOLO11 nano")
    print("  (First time will download ~6MB, please wait...)")
    
    try:
        # Create the AI model object
        # 'yolo11n.pt' means:
        #   - yolo11 = version 11 (latest)
        #   - n = nano (smallest/fastest version)
        #   - .pt = PyTorch format (AI model file type)
        model = YOLO('yolo11n.pt')
        
        print("AI Model loaded successfully!")
        print("This model can detect 80 types of objects")
        print("(people, cars, animals, furniture, etc.)\n")
        
    except Exception as e:
        # If something goes wrong, tell the user
        print(f"ERROR: Could not load model")
        print(f"Reason: {e}")
        print("Make sure you have internet connection for first download")
        return  # Exit the program
    
    
    print("[Step 2/3] Opening camera...")
    
  
    cap = cv2.VideoCapture(0)
    
    # Check if the camera actually opened
    # (Could fail if camera is unplugged or being used by another program)
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        print("\nTroubleshooting tips:")
        print("- Is your USB webcam plugged in?")
        print("- Is another program using the camera?")
        print("- Try unplugging and replugging the camera")
        print("- On Linux, check: ls /dev/video*")
        return  
    
    # Tell the camera what resolution we want (640x480) in this case
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Get the actual camera settings since it could be different
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print("Camera opened successfully!")
    print(f"Resolution: {width} x {height} pixels")
    print(f"Frame rate: {fps} FPS (frames per second)\n")
    
    
    print("[Step 3/3] Starting real-time object detection...")
    print("\n" + "="*60)
    print("SYSTEM IS NOW RUNNING")
    print("="*60)
    print("Keyboard controls:")
    print("'q' = Quit the program")
    print("'s' = Save current frame as image")
    print("'i' = Toggle info display on/off")
    print("="*60 + "\n")
    
    # Variables to keep track of things
    frame_count = 0        # How many frames we've processed
    show_info = True       # Should we show info overlay? (yes/no)
 
    # This main loop runs continuously until pressing 'q'

    
    try:
        while True:   
            ret, frame = cap.read()
            
            # Check if we actually got a frame
            if not ret:
                print("Failed to get frame from camera")
                print("Camera might have disconnected")
                break 
            
            frame_count = frame_count + 1
            
            # this is what FPGA speeds up
            results = model(frame, verbose=False)
            
            annotated_frame = results[0].plot()
    
            if show_info:
                add_info_overlay(annotated_frame, frame_count)
            
            cv2.imshow("Vision System - Press 'q' to quit", annotated_frame)
            
            
            key = cv2.waitKey(1) & 0xFF
            # The "& 0xFF" part is technical - just makes sure we get the right key
            
            # Check which key was pressed
            if key == ord('q'):
                # User pressed 'q' - quit the program
                print("\n  User pressed 'q' - Shutting down...")
                break  # Exit the while loop
            
            elif key == ord('s'):
                # User pressed 's' - save current frame
                filename = f"captured_frame_{frame_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"  ✓ Saved screenshot: {filename}")
            
            elif key == ord('i'):
                # User pressed 'i' - toggle info display
                show_info = not show_info  
                status = "ON" if show_info else "OFF"
                print(f"  Info overlay: {status}")
            
        
    
    except KeyboardInterrupt:
        print("\n\n Program interrupted (Ctrl+C pressed)")
    
    except Exception as e:
        print(f"\n Unexpected error: {e}")
    
    
    finally:
        # This "finally" block ALWAYS runs, even if there was an error
        
        print("\n" + "="*60)
        print("SHUTTING DOWN SYSTEM")
        print("="*60)
        
        cap.release()
        print(f"Camera released")
        
        cv2.destroyAllWindows()
        print(f"Display windows closed")
        
        print(f"\nSession Summary:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Model used: YOLO11 nano")
        
        print("\n" + "="*60)
        print("GOODBYE!")
        print("="*60)

def add_info_overlay(frame, frame_count):
  
    overlay = frame.copy()  

    cv2.rectangle(overlay, (5, 5), (300, 100), (0, 0, 0), -1)
    
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font style
    
    cv2.putText(
        frame,                         
        "System: ACTIVE",               
        (10, 25),                       
        font,                        
        0.5,                      
        (0, 255, 0),                   
        1                           
    )
    
    cv2.putText(frame, f"Frame: {frame_count}", 
               (10, 50), font, 0.5, (0, 255, 0), 1)
    
    cv2.putText(frame, "Model: YOLO11n", 
               (10, 75), font, 0.5, (0, 255, 0), 1)


if __name__ == "__main__":
    main()