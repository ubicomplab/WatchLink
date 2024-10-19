# Created by Anandghan

import numpy as np  # NumPy for numerical operations
import cv2  # OpenCV for image processing
from matplotlib import pyplot as plt  # Matplotlib for plotting

# Import utility functions for reading PDFs and plotting debug information
from utils import pdf_to_images, plot_debug

# Configure matplotlib settings for font and size
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 12


# Function to extract ECG data from the PDF based on pixel information
def extract_ecg_for_pixel(pdf_path, show_debug_output=False):
    images = pdf_to_images(pdf_path)  # Convert PDF pages to images
    image_array = images[0][130:580, :, :]  # Crop region of interest (ROI) from the image

    # Convert the cropped image to grayscale for processing
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to invert the image, making grids black for contour detection
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area in descending order and select the four largest ones
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
    # Further sort the selected contours by their y-coordinate (top to bottom)
    largest_contours = sorted(largest_contours, key=lambda contour: cv2.boundingRect(contour)[1])

    # Define the color range for detecting the red ECG lines in the image
    lower_red = np.array([0, 0, 150])
    upper_red = np.array([150, 150, 255])

    binary_images = []  # List to store binary masks of each grid
    widths = []  # List to store widths of each contour
    ecg = []  # List to store the extracted ECG data

    # Process each detected contour (grid) in the image
    for contour in largest_contours:
        # Get bounding box coordinates (x, y, w, h) for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the region of interest (ROI) from the image, adjusting height to remove the axis
        roi = image_array[y:y + h - 10, x:x + w]

        # Create a mask to isolate red lines within the ROI based on the defined color range
        mask = cv2.inRange(roi, lower_red, upper_red)
        binary_mask = np.where(mask > 0, 1, 0)  # Convert the mask to a binary format (1 for red lines, 0 otherwise)

        # Find the largest y-coordinate for each x-column in the binary mask where the value is non-zero
        y_coords = [binary_mask.shape[0] - np.max(y_indices) for x in range(binary_mask.shape[1])
                    if (y_indices := np.where(binary_mask[:, x] != 0)[0]).size > 0]

        ecg.extend(y_coords)  # Append the y-coordinates to the ECG data list
        binary_images.append(binary_mask)  # Store the binary mask for potential debugging
        widths.append(w)  # Store the width of the contour

        # Optionally, draw rectangles around detected grids for debugging visualization
        if show_debug_output:
            cv2.rectangle(image_array, (x, y), (x + w, y + h - 10), (0, 255, 0), 2)

    # If debugging output is enabled, display the debug plots for the parsed rows
    if show_debug_output:
        plot_debug(image_array, binary_images, widths)

    return np.array(ecg)  # Return the ECG data as a numpy array


# Main section of the code to extract and plot ECG data
if __name__ == "__main__":
    pdf_path = 'pixel_1_Hz.pdf'  # Path to the input PDF file
    parsed_ecg = extract_ecg_for_pixel(pdf_path, show_debug_output=True)  # Extract ECG data with debug visualization

    # Create a time axis for the full ECG plot based on the total number of samples
    time_axis = np.linspace(0, 30, len(parsed_ecg))

    # Plot the ECG data against the time axis
    plt.figure()
    plt.plot(time_axis, parsed_ecg)
    plt.xlabel('Time (s)')  # Label the x-axis as time
    plt.title('Parsed ECG')  # Set the title of the plot
    plt.gca().get_yaxis().set_visible(False)  # Remove y-axis ticks and labels for clarity
    plt.tight_layout()  # Adjust layout for better fit

    plt.show()  # Display the final plot
