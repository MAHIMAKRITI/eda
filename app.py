import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as mcolors


def extract_colors(image_np, num_colors):
    # Reshape the image to be a list of pixels
    pixels = image_np.reshape((-1, 3))

    # Use KMeans to find the most dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    # Get the colors and counts
    colors, counts = np.unique(kmeans.labels_, return_counts=True)

    # Normalize RGB values to the range [0, 1]
    normalized_colors = kmeans.cluster_centers_ / 255.0

    # Get the hex codes of the most common colors
    hex_colors = [mcolors.rgb2hex(color) for color in normalized_colors]

    return hex_colors, counts



def display_color_grid(hex_colors, counts):
    fig, ax = plt.subplots(figsize=(2,2))
    for i, (color, count) in enumerate(zip(hex_colors, counts)):
        ax.add_patch(plt.Rectangle((i % 5, i // 5), 1, 1, fill=True, color=color, edgecolor='black'))
        plt.text(i % 5 + 0.5, i // 5 + 0.5, f"{count}\n{color}", color='black',
                 ha='center', va='center', fontsize=2)

    plt.xlim((0, 5))
    plt.ylim((0, 5))
    plt.axis('off')
    st.pyplot(fig)

def main():
    st.title("Image Color Analyzer App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Convert the uploaded file to a PIL Image
        image = Image.open(uploaded_file)

        # Convert PIL Image to NumPy array
        image_np = np.array(image)

        # Extract colors
        hex_colors, counts = extract_colors(image_np, num_colors=25)

        # Display color grid
        display_color_grid(hex_colors, counts)


if __name__ == "__main__":
    main()
