import cv2
import os
import sys

# --- Configuration ---
INPUT_BASE_DIR = 'images'  # Directory containing celebrity folders
OUTPUT_BASE_DIR = 'cropped_pics' # Directory where cropped faces will be saved
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml' # Path to the downloaded Haar Cascade file
TARGET_SIZE = (224, 224) # Desired output size (width, height) for cropped faces
PADDING_FACTOR = 0.2 
MIN_FACE_SIZE = (30, 30) # Minimum size of face to detect (helps filter noise)
SAVE_JPEG_QUALITY = 95 # Quality for saving JPEG images (0-100, higher is better)



# Check if Haar Cascade file exists
if not os.path.exists(HAAR_CASCADE_PATH):
    print(f"Error: Haar Cascade file not found at '{HAAR_CASCADE_PATH}'")
    print("Please download it from OpenCV's GitHub repository and place it correctly.")
    sys.exit(1)

# Load the Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
if face_cascade.empty():
    print(f"Error: Could not load Haar Cascade classifier from '{HAAR_CASCADE_PATH}'")
    sys.exit(1)

# Check if input directory exists
if not os.path.isdir(INPUT_BASE_DIR):
    print(f"Error: Input directory '{INPUT_BASE_DIR}' not found.")
    sys.exit(1)

# Create the base output directory if it doesn't exist
if not os.path.exists(OUTPUT_BASE_DIR):
    print(f"Creating output directory: '{OUTPUT_BASE_DIR}'")
    os.makedirs(OUTPUT_BASE_DIR)

print("Starting face detection and cropping process...")
print(f"Input Directory:  {os.path.abspath(INPUT_BASE_DIR)}")
print(f"Output Directory: {os.path.abspath(OUTPUT_BASE_DIR)}")
print(f"Target Size:      {TARGET_SIZE}")
print("-" * 30)

# --- Processing ---
skipped_no_face = 0
skipped_multi_face = 0
processed_count = 0
error_count = 0

# Iterate through each celebrity folder in the input directory
for celebrity_name in os.listdir(INPUT_BASE_DIR):
    input_celebrity_dir = os.path.join(INPUT_BASE_DIR, celebrity_name)
    output_celebrity_dir = os.path.join(OUTPUT_BASE_DIR, celebrity_name)

    # Ensure it's actually a directory
    if not os.path.isdir(input_celebrity_dir):
        continue

    print(f"Processing folder: '{celebrity_name}'")

    # Create corresponding output celebrity folder
    if not os.path.exists(output_celebrity_dir):
        os.makedirs(output_celebrity_dir)

    # Iterate through each image file in the celebrity folder
    for filename in os.listdir(input_celebrity_dir):
        input_filepath = os.path.join(input_celebrity_dir, filename)
        output_filepath = os.path.join(output_celebrity_dir, filename)

        # Basic check for image file extensions
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # print(f"  Skipping non-image file: {filename}")
            continue

        try:
            # Load the image in color
            image = cv2.imread(input_filepath)
            if image is None:
                print(f"  Warning: Could not read image: {input_filepath}. Skipping.")
                error_count += 1
                continue

            # Convert the image to grayscale for the detector
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_h, img_w = image.shape[:2]

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray_image,
                scaleFactor=1.1,      # How much the image size is reduced at each image scale
                minNeighbors=5,       # How many neighbors each candidate rectangle should have
                minSize=MIN_FACE_SIZE # Minimum possible object size
            )

            if len(faces) == 0:
                # print(f"  Warning: No faces detected in {filename}. Skipping.")
                skipped_no_face += 1
                continue

            # Handle multiple faces: choose the largest one based on area (w*h)
            if len(faces) > 1:
                # print(f"  Warning: Multiple faces ({len(faces)}) detected in {filename}. Cropping the largest.")
                skipped_multi_face +=1
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                # Keep only the largest face
                faces = [faces[0]]


            # Get coordinates of the (largest) detected face
            (x, y, w, h) = faces[0]

            # Calculate padding
            pad_w = int(w * PADDING_FACTOR / 2)
            pad_h = int(h * PADDING_FACTOR / 2)

            # Calculate coordinates for cropping with padding, ensuring they stay within image bounds
            crop_x1 = max(0, x - pad_w)
            crop_y1 = max(0, y - pad_h)
            crop_x2 = min(img_w, x + w + pad_w)
            crop_y2 = min(img_h, y + h + pad_h)

            # Crop the original color image
            cropped_face = image[crop_y1:crop_y2, crop_x1:crop_x2]

            # Check if crop is valid
            if cropped_face.size == 0:
                print(f"  Warning: Cropped face has zero size for {filename}. Skipping.")
                error_count += 1
                continue

            # Resize the cropped face to the target size with high-quality interpolation
            resized_face = cv2.resize(cropped_face, TARGET_SIZE, interpolation=cv2.INTER_AREA if (w*h > TARGET_SIZE[0]*TARGET_SIZE[1]) else cv2.INTER_CUBIC)

            # Save the resized face
            # For JPEG, add quality parameter
            if output_filepath.lower().endswith(('.jpg', '.jpeg')):
                 cv2.imwrite(output_filepath, resized_face, [cv2.IMWRITE_JPEG_QUALITY, SAVE_JPEG_QUALITY])
            else: # For PNG or others, save normally
                cv2.imwrite(output_filepath, resized_face)

            processed_count += 1

        except Exception as e:
            print(f"  Error processing file {input_filepath}: {e}")
            error_count += 1

# --- Completion Summary ---
print("-" * 30)
print("Processing Complete.")
print(f"Total images processed and saved: {processed_count}")
print(f"Images skipped (no face detected): {skipped_no_face}")
print(f"Images skipped (multiple faces detected, processed largest): {skipped_multi_face}")
print(f"Errors encountered: {error_count}")
print(f"Cropped images saved in: '{os.path.abspath(OUTPUT_BASE_DIR)}'")