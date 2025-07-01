# Load dependancies
import cv2              # OpenCV for video capture and image processing
from PIL import Image   # Pillow for handling image objects and bounding boxes
import numpy as np      # NumPy for working with arrays and numerical operations

# capture video input
cap = cv2.VideoCapture('/dev/video10')

# Function to get HSV color range based on a BGR input color
def get_limits(color):
    '''
    Convert BGR color to a 1x1 pixel array (1,1,3) which opencv accepts as an image for color conversion
    we use np.unit8 - unassigned interger with 8 bits which can store 256 values creating our pixel range 
    from 0 to 255
    '''
    c = np.uint8([[color]])                        
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)     

    # Define lower and upper HSV range with a ±10 hue tolerance and fixed saturation and value numbers.
    lower_limit = hsvC[0][0][0] - 10, 100, 100
    upper_limit = hsvC[0][0][0] + 10, 255, 255

    # convert to numpy arrays for masking
    lower_limit = np.array(lower_limit, dtype=np.uint8)
    upper_limit = np.array(upper_limit, dtype=np.uint8)

    return lower_limit, upper_limit

# define color in bgr
yellow = [0, 255, 255]

while True:
    # Read a frame from the video capture
    ret, frame = cap.read() # ret is a boolean (success), frame is the image

    # Convert frames to hsv color space
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the lower and upper HSV limits for the selected color
    lower_limit, upper_limit = get_limits(yellow)

    # Create a binary mask where the target color is white and everything else is black
    mask = cv2.inRange(hsv_img, lower_limit, upper_limit)

    # Convert mask (NumPy array) to Pillow image
    mask_ = Image.fromarray(mask)

    # Get the bounding box of the white area in the mask
    bbox= mask_.getbbox()

    # If a region with the target color is found, draw a red rectangle around it
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    # Show the result in a window
    cv2.imshow('frame', frame)

    '''
    cv2.waitkey(1) tells opencv to wait for 1 millisecond for a key event, if no key is pressed it returns -1.
    If a key is pressed, it returns tha ASCII value of the key, hence & 0xFF, a bitwise AND operation keeps the last 8 bits returned by cv2.waitKey
    ord('q) returns the ASCII interger for q
    '''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

'''
After exiting the progam by pressing q, cap.release() releases the video capture object making it now useable by 
other applications or programs like ZOOM and OBS.
cv2.destroyAllWindows closes any windows that were opened using cv2.imshow()
'''
cap.release()
cv2.destroyAllWindows()