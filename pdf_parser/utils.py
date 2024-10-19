# Created by Anandghan

import fitz  # PyMuPDF library for handling PDF files
import numpy as np
from matplotlib import pyplot as plt

# Set the font family and size for plots
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 12


# Function to read a PDF and convert each page to an image as a numpy array
def pdf_to_images(file_path):
    # Open the PDF document
    pdf_document = fitz.open(file_path)

    # Convert each page of the PDF to a numpy array (image)
    images = [np.frombuffer(page.get_pixmap().samples, dtype=np.uint8).reshape(
        page.get_pixmap().height, page.get_pixmap().width, page.get_pixmap().n)
        for page in pdf_document]

    # Close the PDF after processing
    pdf_document.close()

    return images  # Return the list of images


# Function to plot the debug information for each parsed row, starting x from the previous row's max x
def plot_debug(image_array, binary_images, widths):
    plt.imshow(image_array)  # Display the main image as a background
    plt.title('PDF View')  # Set the title of the plot
    plt.axis('off')  # Hide the axes for a cleaner view

    # Calculate the total width of all rows combined to normalize durations
    total_width = sum(widths)
    total_duration = 30  # Assume a total duration of 30 seconds for the entire image

    # Create a new figure with the size adjusted based on the number of binary images (rows)
    plt.figure(figsize=(6, len(binary_images) * 2))

    # Variable to track the cumulative starting point on the x-axis
    cumulative_x_start = 0

    # Iterate through each binary image (grid) and its corresponding width
    for i, (binary_image, width) in enumerate(zip(binary_images, widths), 1):
        # Extract y-coordinates for the current binary image where non-zero values exist
        y_coords = [binary_image.shape[0] - np.max(y_indices) for x in range(binary_image.shape[1])
                    if (y_indices := np.where(binary_image[:, x] != 0)[0]).size > 0]

        # Calculate the time span for the current row based on its width relative to the total width
        row_duration = (width / total_width) * total_duration

        # Create a time axis starting from the cumulative x start for the row duration
        time_axis = np.linspace(cumulative_x_start, cumulative_x_start + row_duration, len(y_coords))

        # Create a subplot for the current row
        plt.subplot(len(binary_images), 1, i)
        plt.plot(time_axis, y_coords)  # Plot y-coordinates against the time axis
        plt.ylim((0, 100))  # Set y-axis limits
        plt.xlabel('Time (s)')  # Label the x-axis
        plt.title(f'Parsed Row {i}')  # Set the title for the current row

        # Remove y-axis ticks and labels for clarity
        plt.gca().get_yaxis().set_visible(False)

        # Update cumulative_x_start for the next row to start where the current row ends
        cumulative_x_start += row_duration

    plt.tight_layout()  # Adjust subplot parameters to fit the figure area neatly
