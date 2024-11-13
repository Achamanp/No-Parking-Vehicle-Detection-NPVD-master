import cv2
import numpy as np
import os
import re
import DetectChars
import DetectPlates
import PossiblePlate
import csv
from datetime import datetime
import time
from pathlib import Path

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

class DataDisplay:
    def __init__(self, window_name="License Plate Data"):
        self.window_name = window_name
        self.data_history = []
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 600, 400)
        
    def update(self, plate_number, is_no_parking):
        if plate_number:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.data_history.append((plate_number, timestamp, is_no_parking))
            if len(self.data_history) > 10:  # Keep only last 10 entries
                self.data_history.pop(0)
        
        # Create blank canvas for data display
        height = 400
        width = 600
        display = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add title
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, "Detected License Plates", (20, 40), font, 1, SCALAR_WHITE, 2)
        
        # Add header
        cv2.putText(display, "Plate Number", (20, 80), font, 0.7, SCALAR_YELLOW, 2)
        cv2.putText(display, "Timestamp", (300, 80), font, 0.7, SCALAR_YELLOW, 2)
        cv2.putText(display, "In No-Parking Zone", (450, 80), font, 0.7, SCALAR_YELLOW, 2)
        
        # Draw separator line
        cv2.line(display, (20, 90), (width-20, 90), SCALAR_WHITE, 1)
        
        # Add data entries
        y_pos = 120
        for plate, time, is_no_parking in self.data_history:
            cv2.putText(display, plate, (20, y_pos), font, 0.7, SCALAR_WHITE, 1)
            cv2.putText(display, time, (300, y_pos), font, 0.7, SCALAR_WHITE, 1)
            if is_no_parking:
                cv2.putText(display, "Yes", (450, y_pos), font, 0.7, SCALAR_RED, 1)
            else:
                cv2.putText(display, "No", (450, y_pos), font, 0.7, SCALAR_GREEN, 1)
            y_pos += 30
        
        # Add total count
        cv2.putText(display, f"Total Detections: {len(self.data_history)}", 
                   (20, height - 20), font, 0.7, SCALAR_GREEN, 2)
        
        cv2.imshow(self.window_name, display)
    
    def close(self):
        cv2.destroyWindow(self.window_name)

class OutputDisplay:
    def __init__(self, window_name="License Plate Detection"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
    def create_side_by_side_display(self, original_frame, processed_frame, plate_number=None, is_no_parking=False):
        # Get frame dimensions
        height = max(original_frame.shape[0], processed_frame.shape[0])
        width = original_frame.shape[1] + processed_frame.shape[1]
        
        # Create blank canvas
        display = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Copy original frame to left side
        display[:original_frame.shape[0], :original_frame.shape[1]] = original_frame
        
        # Copy processed frame to right side
        display[:processed_frame.shape[0], original_frame.shape[1]:] = processed_frame
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, "Original", (10, 30), font, 1, SCALAR_WHITE, 2)
        cv2.putText(display, "Processed", (original_frame.shape[1] + 10, 30), font, 1, SCALAR_WHITE, 2)
        
        # Add plate number and parking zone info if detected
        if plate_number:
            if is_no_parking:
                cv2.putText(display, f"Detected Plate: {plate_number} (No-Parking Zone)", 
                           (10, height - 20), font, 0.75, SCALAR_RED, 2)
            else:
                cv2.putText(display, f"Detected Plate: {plate_number}", 
                           (10, height - 20), font, 0.75, SCALAR_GREEN, 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(display, timestamp, (width - 250, height - 20), 
                   font, 0.75, SCALAR_WHITE, 2)
        
        return display

    def show(self, display):
        cv2.imshow(self.window_name, display)
    
    def close(self):
        cv2.destroyWindow(self.window_name)

def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)
    p2fRectPoints = np.int32(p2fRectPoints)
    
    cv2.line(imgOriginalScene, (p2fRectPoints[0][0], p2fRectPoints[0][1]), 
             (p2fRectPoints[1][0], p2fRectPoints[1][1]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, (p2fRectPoints[1][0], p2fRectPoints[1][1]), 
             (p2fRectPoints[2][0], p2fRectPoints[2][1]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, (p2fRectPoints[2][0], p2fRectPoints[2][1]), 
             (p2fRectPoints[3][0], p2fRectPoints[3][1]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, (p2fRectPoints[3][0], p2fRectPoints[3][1]), 
             (p2fRectPoints[0][0], p2fRectPoints[0][1]), SCALAR_RED, 2)

def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    center = np.int32(licPlate.rrLocationOfPlateInScene[0])
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    
    (text_width, text_height), baseline = cv2.getTextSize(licPlate.strChars, font, font_scale, thickness)
    text_x = center[0] - (text_width // 2)
    text_y = center[1] + (text_height // 2)
    
    cv2.putText(imgOriginalScene, licPlate.strChars, (text_x, text_y), 
                font, font_scale, SCALAR_YELLOW, thickness)

def check_parking_zone(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through the contours and check for parking zone markers
    for cnt in contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Check if the contour is a long, thin shape (likely a parking zone marker)
        if w > 5 * h or h > 5 * w:
            # Check if the contour is located in the lower half of the frame (likely on the ground)
            if y > frame.shape[0] // 2:
                # Check if the contour is near the edge of the frame (likely on the road)
                if x < 50 or x > frame.shape[1] - 50:
                    return True
    
    # No no-parking zone detected
    return False

def draw_parking_zones(frame, parking_zones, no_parking_zones):
    # Draw parking zones
    for x, y, w, h in parking_zones:
        cv2.rectangle(frame, (x, y), (x + w, y + h), SCALAR_GREEN, 2)
    
    # Draw no-parking zones
    for x, y, w, h in no_parking_zones:
        cv2.rectangle(frame, (x, y), (x + w, y + h), SCALAR_RED, 2)
    
    return frame
def process_frame(frame, blnKNNTrainingSuccessful):
    if not blnKNNTrainingSuccessful:
        print("\nerror: KNN training was not successful\n")
        return frame, None, False

    imgOriginalScene = frame.copy()

    if imgOriginalScene is None:
        print("\nerror: image not read from file \n")
        return frame, None, False

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)
    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

    if len(listOfPossiblePlates) == 0:
        print("\nno license plates were detected\n")
        return frame, None, False
    else:
        listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)
        licPlate = listOfPossiblePlates[0]

        if len(licPlate.strChars) == 0:
            print("\nno characters were detected\n")
            return frame, None, False

        try:
            drawRedRectangleAroundPlate(imgOriginalScene, licPlate)
            writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)
        except Exception as e:
            print(f"Error drawing on frame: {str(e)}")
            return frame, licPlate.strChars, check_parking_zone(frame)

        print("\nlicense plate read from frame = " + licPlate.strChars + "\n")
        
        if showSteps:
            cv2.imshow("imgPlate", licPlate.imgPlate)
            cv2.imshow("imgThresh", licPlate.imgThresh)

        is_no_parking = check_parking_zone(frame)
        return imgOriginalScene, licPlate.strChars, is_no_parking

def save_detection(processed_frame, plate_number, is_no_parking):
    timestamp = datetime.now()
    date_str = timestamp.strftime('%Y_%m_%d')
    time_str = timestamp.strftime('%H_%M_%S')
    filename = f"{date_str}_{time_str}"
    
    # Create directories if they don't exist
    os.makedirs('logs/images', exist_ok=True)
    
    # Save image
    cv2.imwrite(f"logs/images/{filename}.jpg", processed_frame)
    
    # Save to log file
    log_file = "logs/log.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, "a+") as file:
        file.write(f"{filename}.jpg,{plate_number},{date_str},{time_str},{int(is_no_parking)}\n")
    
    print(f"Saved new plate: {plate_number} (In No-Parking Zone: {is_no_parking})")

def process_image(image_path, blnKNNTrainingSuccessful):
    print(f"Processing image: {image_path}")
    original_frame = cv2.imread(image_path)
    
    if original_frame is None:
        print("Error: Could not read image file")
        return
    
    display = OutputDisplay("Image Detection")
    data_display = DataDisplay("License Plate Data")
    processed_frame, plate_number, is_no_parking = process_frame(original_frame.copy(), blnKNNTrainingSuccessful)
    
    if plate_number:
        save_detection(processed_frame, plate_number, is_no_parking)
        data_display.update(plate_number, is_no_parking)
    
    if processed_frame is not None:
        combined_display = display.create_side_by_side_display(
            original_frame, processed_frame, plate_number, is_no_parking)
        display.show(combined_display)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    display.close()
    data_display.close()

def process_video(video_path, blnKNNTrainingSuccessful):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    processed_plates = set()
    frame_count = 0
    display = OutputDisplay("Video Detection")
    data_display = DataDisplay("License Plate Data")
    last_combined_display = None  # Store the last display state

    while frame_count < total_frames:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video file")
            break

        frame_count += 1
        if frame_count % 3 != 0:  # Process every 3rd frame
            continue

        try:
            processed_frame, plate_number, is_no_parking = process_frame(frame.copy(), blnKNNTrainingSuccessful)
            
            if plate_number and plate_number not in processed_plates:
                processed_plates.add(plate_number)
                save_detection(processed_frame, plate_number, is_no_parking)
                data_display.update(plate_number, is_no_parking)
                print(f"Processed {len(processed_plates)} unique plates out of {frame_count}/{total_frames} frames")

            if processed_frame is not None:
                last_combined_display = display.create_side_by_side_display(
                    frame, processed_frame, plate_number, is_no_parking)
            else:
                last_combined_display = display.create_side_by_side_display(
                    frame, frame, None, False)
                
            display.show(last_combined_display)
            
            # Update data display even if no new plate is detected
            if plate_number:
                data_display.update(plate_number, is_no_parking)

        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            last_combined_display = display.create_side_by_side_display(frame, frame, None, False)
            display.show(last_combined_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing stopped by user")
            break

        # If we've processed all unique plates (10 in your case), stop processing
        if len(processed_plates) >= 10:
            print("All expected plates have been detected. Stopping processing.")
            break

    # Print summary
    print(f"\nProcessing complete:")
    print(f"Total frames processed: {frame_count}")
    print(f"Unique plates detected: {len(processed_plates)}")
    print("Detected plates:", processed_plates)

    # Release the video capture
    cap.release()

    # Keep displaying the last frame until user closes
    print("\nPress 'Q' to exit...")
    while True:
        if last_combined_display is not None:
            display.show(last_combined_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Clean up windows
    display.close()
    data_display.close()

def main(input_path):
    print("License Plate Detection System Started")
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('logs/images', exist_ok=True)

    # Load KNN model
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()

    if not blnKNNTrainingSuccessful:
        print("\nerror: KNN training was not successful\n")
        return

    if not os.path.exists(input_path):
        print(f"Error: File {input_path} does not exist")
        return

    # Determine file type and process accordingly
    file_extension = Path(input_path).suffix.lower()
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}

    if file_extension in image_extensions:
        process_image(input_path, blnKNNTrainingSuccessful)
    elif file_extension in video_extensions:
        process_video(input_path, blnKNNTrainingSuccessful)
    else:
        print(f"Error: Unsupported file type {file_extension}")
        print(f"Supported image formats: {', '.join(image_extensions)}")
        print(f"Supported video formats: {', '.join(video_extensions)}")

if __name__ == "__main__":
   main("D:\\No-Parking-Vehicle-Detection-NPVD-master\\LicPlateImages\\6.jpg")
