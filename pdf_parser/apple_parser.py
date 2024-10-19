# Created by Anandghan

import cv2  # OpenCV library for image processing
import numpy as np  # NumPy for numerical operations
from matplotlib import pyplot as plt  # Matplotlib for plotting

# Import utility functions for reading PDFs and plotting debug information
from utils import pdf_to_images, plot_debug

# Set global font properties for plots
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 12


# Function to extract ECG data from an Apple Watch ECG PDF
def extract_ecg_for_apple(pdf_path, show_debug_output=False):
    images = pdf_to_images(pdf_path)  # Convert PDF pages to images
    image_array = images[0][170:530, :, :]  # Crop the region of interest (ROI) directly from the first page

    # Convert the cropped image to grayscale for processing
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Apply threshold to isolate grid lines (invert binary for easier contour detection)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the grids in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area in descending order and select the three largest
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    # Further sort the selected contours by their y-coordinate (top to bottom)
    largest_contours = sorted(largest_contours, key=lambda contour: cv2.boundingRect(contour)[1])

    # Define the color range for detecting the red ECG signal in the cropped image
    lower_red = np.array([150, 0, 0])
    upper_red = np.array([255, 150, 150])

    ecg = []  # List to store y-coordinates of the ECG signal across the image
    binary_images = []  # List to store binary masks of each detected grid
    widths = []  # List to store the widths of each detected grid

    # Iterate through each of the sorted largest contours
    for contour in largest_contours:
        x, y, w, h = cv2.boundingRect(contour)  # Get the bounding box coordinates and dimensions
        h -= 8  # Adjust height to remove the axis area

        # Check if the ROI is within the bounds of the image
        if y >= 0 and y + h <= image_array.shape[0] and x >= 0 and x + w <= image_array.shape[1]:
            roi = image_array[y:y + h, x:x + w]  # Extract the region of interest (grid)

            # Apply color mask to detect red color and convert it to a binary mask
            binary_mask = cv2.inRange(roi, lower_red, upper_red) > 0

            # Extract y-coordinates where the binary mask is non-zero for each column (x)
            y_coords = [binary_mask.shape[0] - np.max(y_indices)
                        for x in range(binary_mask.shape[1])
                        if (y_indices := np.where(binary_mask[:, x] != 0)[0]).size > 0]

            ecg.extend(y_coords)  # Append y-coordinates to the ECG list
            binary_images.append(binary_mask)  # Store the binary mask for debug visualization
            widths.append(w)  # Store the width of the grid

            # Draw a rectangle around the detected grid for debugging purposes
            if show_debug_output:
                cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # If debugging output is enabled, display the debug plot
    if show_debug_output:
        plot_debug(image_array, binary_images, widths)

    return np.array(ecg)  # Return the extracted ECG signal as a numpy array


if __name__ == "__main__":
    debug = False  # Set debugging mode

    pdf_path = 'apple_1_Hz.pdf'  # Path to the PDF file
    label = '1Hz'  # Label for the ECG data (optional)

    # Extract ECG signal from the PDF
    ecg_signal = extract_ecg_for_apple(pdf_path, show_debug_output=True)

    # Create a time axis for the ECG signal based on a total duration of 30 seconds
    time_axis = np.linspace(0, 30, len(ecg_signal))

    plt.figure()  # Create a new figure for the ECG plot
    plt.plot(time_axis, ecg_signal)  # Plot the ECG signal against the time axis
    plt.xlabel('Time (s)')  # Label the x-axis
    plt.title('Parsed ECG')  # Set the title of the plot
    plt.gca().get_yaxis().set_visible(False)  # Remove y-axis ticks and labels for clarity
    plt.tight_layout()  # Adjust layout to fit the plot area neatly

    plt.show()  # Display the plot
