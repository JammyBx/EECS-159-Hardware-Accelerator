"""
This script prepares the Raspberry Pi for the vision system.
Run this ONCE before using the vision system.

1. Checks your Python version
2. Installs required software packages
3. Tests if your camera works
4. Downloads a sample image for testing
"""
import subprocess  
import sys       
import os       

# 1.
def check_python_version():

    print("Checking Python version...")
    
    version = sys.version_info
    major = version.major  
    minor = version.minor 
    micro = version.micro  
    
    print(f"  Current version: Python {major}.{minor}.{micro}")
    
    # Check if version is too old
    if major < 3:
        print("ERROR: Need Python 3.x")
        return False
    if major == 3 and minor < 8:
        print(f"ERROR: Need Python 3.8 or newer (you have 3.{minor})")
        return False
    
    print("Python version is OK\n")
    return True

# 2. 
def install_dependencies():
    
    # Install required Python packages
    # We need opencv-python, ultralytics, numpy
    print("Installing required packages...")
    print("This will take a few minutes. Please be patient!\n")
    
    packages = [
        'opencv-python',    # OpenCV library
        'ultralytics',      # YOLO11 library
        'onnx',             # ONNX model format (for FPGA pipeline)
        'onnxruntime',      # ONNX inference engine (for layer-by-layer execution)
        'pyserial',         # UART serial communication (for FPGA data transfer)
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        print("-" * 40)
        
        try:
          
            subprocess.check_call([
                sys.executable,      
                "-m", "pip",      
                "install",          
                "--upgrade",        
                package             
            ])
            print(f"{package} installed successfully\n")
            
        except subprocess.CalledProcessError as e:
            # If installation failed
            print(f"Failed to install {package}")
            print(f"  Error: {e}\n")
            return False
    
    print("All packages installed!\n")
    return True

# 3. 
def test_camera():
 
    print("Testing camera connection...")
    
    # Import OpenCV (must be installed first)
    import cv2
    
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera detected!")
        print(f"Resolution: {width} x {height}")
        
        cap.release()
        return True
    else:
        # No camera found
        print("No camera detected")
        print("This is OK - you can still test with static images")
        print("Connect USB webcam later for live detection\n")
        return False

# 4. 
def download_sample_image():
 
    print("\nDownloading sample test image...")
    
    try:
        import urllib.request
        
        # THIS IS A SAMPLE IMAGE (bus with people)
        url = "https://ultralytics.com/images/bus.jpg"
        filename = "test_image.jpg"
        
        if os.path.exists(filename):
            print(f"Test image already exists: {filename}")
            return True
        
        print("Downloading... (may take a moment)")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded: {filename}")
        print(f"(You can use this to test without a camera)\n")
        return True
        
    except Exception as e:
        print(f"Failed to download sample image")
        print(f"Error: {e}")
        print("Don't worry - you can use your own image instead\n")
        return False


def main():

    print("\n" + "="*60)
    print("RASPBERRY PI SETUP FOR VISION SYSTEM")
    print("="*60)
    print("This will prepare your Pi for object detection")
    print("="*60 + "\n")
    
    if not check_python_version():
        print("Setup failed: Python version too old")
        print("Please upgrade Python to 3.8 or newer")
        return
    
    print("="*60)
    if not install_dependencies():
        print("Setup failed during package installation")
        print("Check your internet connection and try again")
        return
    
    print("="*60)
    camera_ok = test_camera()
    
    print("="*60)
    download_sample_image()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("Python version OK")
    print("Required packages installed")
    print("Sample test image ready")
    
    if camera_ok:
        print("Camera detected and working")
    else:
        print("No camera detected (can test with images)")
    
    print("\nNext steps:")
    print("1. Test with sample image:")
    print("python3 vision_system_static.py")
    print("\n2. Test with live camera:")
    print("python3 vision_system.py")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()