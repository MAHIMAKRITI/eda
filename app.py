import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as mcolors

def extract_colors(image_np, num_colors):
    pixels = image_np.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)

    colors, counts = np.unique(kmeans.labels_, return_counts=True)
    normalized_colors = kmeans.cluster_centers_ / 255.0
    hex_colors = [mcolors.rgb2hex(color) for color in normalized_colors]

    return hex_colors, counts

def kmeans_segmentation(image_np, num_colors):
    pixels = image_np.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    segmented_image = np.reshape(kmeans.labels_, image_np.shape[:2])
    return segmented_image

def display_color_grid(hex_colors, counts):
    # Sort hex_colors and counts based on counts in descending order
    sorted_indices = np.argsort(counts)[::-1]
    hex_colors = [hex_colors[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]

    num_colors = len(hex_colors)
    num_rows = (num_colors + 4) // 5  # Display 5 colors per row

    fig, ax = plt.subplots(num_rows, 5, figsize=(10, 2 * num_rows))

    for i in range(num_rows):
        for j in range(5):
            index = i * 5 + j
            if index < num_colors:
                color = hex_colors[index]
                count = counts[index]
                
                # Convert hex code to RGB
                rgb_color = mcolors.hex2color(color)
                rgb_color = tuple(int(val * 255) for val in rgb_color)
                
                ax[i, j].add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, color=color, edgecolor='black'))
                ax[i, j].text(0.5, 0.5, f"{count}\n{color}\n{rgb_color}", color='black',
                              ha='center', va='center', fontsize=8)
            ax[i, j].axis('off')

    plt.tight_layout()
    st.pyplot(fig)



def display_segmentation(original_image, segmented_image):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(segmented_image, cmap='viridis')
    axes[1].set_title('K-means Segmentation')
    axes[1].axis('off')

    # Add text description below the segmented image
    # description = "K-means Segmentation assigns each pixel to a cluster based on color similarity."
    # axes[1].text(0, -0.1, description, ha='left', va='center', transform=axes[1].transAxes, fontsize=10)

    st.pyplot(fig)


def main():
    st.set_page_config(page_title="Palette Analyzer")
    st.title("Image Palette Analyzer")
    st.write("This app allows you to analyze the color palette of an image.")
    
    # Upload image and set the number of colors
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    num_colors = st.number_input("Number of Colors to Extract", min_value=1, max_value=50, value=15)

    # Toggle button for displaying K-means segmented image
    show_kmeans_segmentation = st.checkbox("Show K-means Segmentation", False)

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        image = Image.open(uploaded_file)
        image_np = np.array(image)

        hex_colors, counts = extract_colors(image_np, num_colors)
        
        # Display color palette
        display_color_grid(hex_colors, counts)

        # Conditionally display K-means segmented image based on toggle button
        if show_kmeans_segmentation:
            segmented_image = kmeans_segmentation(image_np, num_colors)
            st.write("K-means clustering is applied to the image to group similar colors together. "
                     "Each segment represents a cluster of colors. Change the number of colors you want to extract to see how the clustering changes")
            display_segmentation(image_np, segmented_image)
            # st.write("This app allows you to analyze the color palette of an image using K-means clustering.")

    
if __name__ == "__main__":
    main()
