import ast
import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length=50):
    """
    Draws corner borders similar to the style shown in the provided example.
    """
    x1, y1 = map(int, top_left)
    x2, y2 = map(int, bottom_right)

    # Corner lines length
    line_length_x = line_length
    line_length_y = line_length

    # Top-left corner
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)

    # Top-right corner
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    # Bottom-left corner
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)

    # Bottom-right corner
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)

    return img


def get_license_crop(frame, bbox):
    """
    Crop and resize the license plate from the frame based on the bounding box coordinates.
    """
    x1, y1, x2, y2 = map(int, bbox)
    license_crop = frame[y1:y2, x1:x2]

    # Handle the case where the bounding box is out of frame bounds
    if license_crop.size == 0:
        return np.zeros((200, 200, 3), dtype=np.uint8)  # Return a blank image if out of bounds

    aspect_ratio = (x2 - x1) / (y2 - y1)  # Compute aspect ratio
    new_width = int(200 * aspect_ratio)  # Resize width to maintain aspect ratio
    return cv2.resize(license_crop, (new_width, 200))


def overlay_license_plate(frame, license_crop, license_text, overlay_position):
    """
    Overlay the cropped license plate and text on the frame at a given position.
    """
    H, W, _ = license_crop.shape
    overlay_x1, overlay_y1 = overlay_position
    overlay_x2 = overlay_x1 + W
    overlay_y2 = overlay_y1 + H

    # Ensure overlay doesn't go out of frame boundaries
    overlay_x1 = max(0, overlay_x1)
    overlay_y1 = max(0, overlay_y1)
    overlay_x2 = min(frame.shape[1], overlay_x2)
    overlay_y2 = min(frame.shape[0], overlay_y2)

    # Make sure the overlay region is valid (width and height should be > 0)
    if overlay_x2 > overlay_x1 and overlay_y2 > overlay_y1:
        # Resize license_crop if necessary to fit the available area
        license_crop_resized = cv2.resize(license_crop, (overlay_x2 - overlay_x1, overlay_y2 - overlay_y1))

        # Overlay the license plate image at the provided position
        frame[overlay_y1:overlay_y2, overlay_x1:overlay_x2] = license_crop_resized

        # Draw a white background for the text above the overlay
        cv2.rectangle(frame, (overlay_x1, overlay_y1 - 30), (overlay_x2, overlay_y1), (255, 255, 255), -1)

        # Add license plate number text at the provided position
        text_size = cv2.getTextSize(license_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (overlay_x1 + overlay_x2 - text_size[0]) // 2
        text_y = overlay_y1 - 10  # Position above the overlay

        cv2.putText(frame, license_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    else:
        print(f"Skipping overlay due to invalid region dimensions: ({overlay_x1}, {overlay_y1}) to ({overlay_x2}, {overlay_y2})")




def process_video(input_video_path, output_video_path, results):
    """
    Process the video, overlaying license plates and text for each car while avoiding overlaps.
    """
    # Load video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Prepare license plate information per car
    license_plate_info = {}
    for car_id in np.unique(results['car_id']):
        best_row = results[results['car_id'] == car_id].nlargest(1, 'license_number_score').iloc[0]
        frame_idx = best_row['frame_nmr']
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, frame = cap.read()

        license_bbox = ast.literal_eval(
            best_row['license_plate_bbox'].replace('[ ', '[').replace('  ', ' ').replace(' ', ','))
        license_crop = get_license_crop(frame, license_bbox)

        license_plate_info[car_id] = {
            'license_crop': license_crop,
            'license_plate_number': best_row['license_number']
        }

    # Process video frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_nmr = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_nmr += 1

        current_frame_data = results[results['frame_nmr'] == frame_nmr]
        overlay_offset = 0  # Initialize an offset to move license plates vertically

        for _, row in current_frame_data.iterrows():
            car_bbox = ast.literal_eval(row['car_bbox'].replace('[ ', '[').replace('  ', ' ').replace(' ', ','))
            license_bbox = ast.literal_eval(
                row['license_plate_bbox'].replace('[ ', '[').replace('  ', ' ').replace(' ', ','))

            # Draw car and license plate bounding boxes
            draw_border(frame, car_bbox[:2], car_bbox[2:], color=(0, 255, 0), thickness=25)
            cv2.rectangle(frame, tuple(map(int, license_bbox[:2])), tuple(map(int, license_bbox[2:])), (0, 0, 255), 12)

            # Calculate overlay position for each license plate
            overlay_x1 = 50  # Fixed x-position for the overlay
            overlay_y1 = frame.shape[0] - 400 - 50 - overlay_offset  # Varying y-position based on offset
            overlay_position = (overlay_x1, overlay_y1)

            # Increment the offset for the next license plate
            overlay_offset += 450  # Adjust vertical spacing between overlays

            # Overlay license plate and text at the calculated position
            license_data = license_plate_info[row['car_id']]
            overlay_license_plate(frame, license_data['license_crop'], license_data['license_plate_number'], overlay_position)

        out.write(frame)

    # Release video writer and capture
    out.release()
    cap.release()


# Load interpolated results from CSV
results = pd.read_csv('./test_interpolated.csv')
results['license_number_score'] = pd.to_numeric(results['license_number_score'], errors='coerce')

# Process the video and save the output
process_video(
    input_video_path=r"C:\Users\moh19\Downloads\nr.mp4",
    output_video_path='./out.mp4',
    results=results
)
