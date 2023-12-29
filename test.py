import cv2
import numpy as np


def remove_white_edges(img_path):
    # Read the binary image
    binary_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Invert the binary image (white to black, black to white)
    inverted_img = cv2.bitwise_not(binary_img)

    # Apply dilation to fill small gaps
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(inverted_img, kernel, iterations=1)

    # Apply erosion to remove edges
    eroded_img = cv2.erode(dilated_img, kernel, iterations=1)

    # Invert the result back to the original orientation
    result_img = cv2.bitwise_not(eroded_img)

    # Check the color space
    if result_img.ndim == 2:
        print("Color Space: Grayscale")
    elif result_img.ndim == 3:
        print("Color Space: BGR (Color)")
    else:
        print("Unknown Color Space")

    # Display the result
    cv2.imshow("Original Binary Image", binary_img)
    cv2.imshow("Processed Image", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
remove_white_edges("./dataset/dirty.png")
