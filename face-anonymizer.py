import cv2
import argparse
import mediapipe as mp 

def process_img(img, face_detection):
    """
    Process an image to detect faces and blur them.
    
    Args:
        img: Input image in BGR format (OpenCV default)
        face_detection: MediaPipe face detection model instance
        
    Returns:
        Image with faces blurred
    """
    # Convert image from BGR (OpenCV format) to RGB (MediaPipe format)    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe face detection
    out = face_detection.process(img_rgb)

    # Get image dimensions for converting relative coordinates to absolute
    h_, w_, _ = img.shape     
 
    # Check if any faces were detected
    if out.detections is not None:
        # Process each detected face
        for detection in out.detections:

            # Extract location data from the detection
            location_data = detection.location_data

            # Get the bounding box in relative coordinates (0-1)
            bbox = location_data.relative_bounding_box

            # Extract bounding box coordinates and dimensio
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            # Convert relative coordinates to absolute pixel values
            x1 = int(x1 * w_)
            y1 = int(y1 * h_)
            w  = int(w * w_)
            h  = int(h * h_)                                    

            # Apply blur effect to the face region using a 50x50 kernel
            img[y1:y1+h, x1:x1+w, :] = cv2.blur(img[y1:y1+h, x1:x1+w, :], (50,50))    

    return img

# Set up command line argument parser
args = argparse.ArgumentParser()

# Define command-line arguments with default values
args.add_argument('--mode', default='webcam')    # Operating mode: image, video, or webcam
args.add_argument('--filePath', default=None)    # Path to input file (for image or video mode)
args = args.parse_args()


# Initialize MediaPipe face detection module
mp_face_detection = mp.solutions.face_detection

# Create face detection model with context manager for proper resource handling
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    # IMAGE MODE: Process a single image file
    if args.mode in ['image']:
        # Read the input image
        img = cv2.imread(args.filePath)       

        # Process image to blur faces
        img = process_img(img, face_detection)

        # save image
        img_out_path = '/windows/Wallpapers/Mine/IMG_20220513_160821-resized-out.jpg'
        cv2.imwrite(img_out_path, img)

    # VIDEO MODE: Process a video file
    elif args.mode in ['video']:
        # Open the video file
        cap = cv2.VideoCapture(args.filePath)
        # Read the first frame
        ret, frame = cap.read()                  

        # Set up video writer for saving output
        video_out_path = '/windows/Socials/YouTube/Bollinger Bands Video/Phone/VID_20230712_162016-out.mp4'
        output_video = cv2.VideoWriter(video_out_path,
                        cv2.VideoWriter_fourcc(*'MP4V'),   # Video codec
                        25,                                # Frame rate (FPS)
                        (frame.shape[1], frame.shape[0]))  # Frame dimensions

        # Process video frame by frame
        while ret:
            # Process current frame to blur faces
            frame = process_img(frame, face_detection)
            # Write processed frame to output video
            output_video.write(frame)
             # Read the next frame
            ret, frame = cap.read()

        # Release resources    
        cap.release()
        output_video.release()
    
    # WEBCAM MODE: Process live webcam feed    
    elif args.mode in ['webcam']:
        # Device ID for the webcam
        cap = cv2.VideoCapture('/dev/video10')
        # Read the first frame
        ret, frame = cap.read()

        # Process webcam stream frame by frame
        while ret:
            # Process current frame to blur faces
            frame = process_img(frame, face_detection)

            # Display the processed frame
            cv2.imshow('frame', frame)

            # Check every millisecond for the quit command from the keyboard
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Read the next frame
            ret, frame = cap.read()

        # Release webcam resources
        cap.release()


