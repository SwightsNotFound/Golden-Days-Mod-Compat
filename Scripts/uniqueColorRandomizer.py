import random
from PIL import Image

def identify_and_replace_colors(image_path, output_path):
    # Open the image
    image = Image.open(image_path)

    # Get the image pixels
    pixels = list(image.getdata())

    # Create a dictionary to store mapping of old colors to new randomized colors
    color_mapping = {}

    # Iterate through each pixel
    for pixel in pixels:
        color = pixel[:3]  # Extract RGB values
        transparency = pixel[3]  # Extract transparency value (alpha channel)

        # If the color is not in the mapping, generate a new randomized color
        if color not in color_mapping:
            # new_color = (random.randint(0, 63), random.randint(0, 63), random.randint(0, 63))
            new_color = (pixel[0]+63, pixel[1]-63, pixel[2]-63)
            color_mapping[color] = new_color

    # Create a new image with replaced colors
    new_image = Image.new("RGBA", image.size)
    new_pixels = [color_mapping[pixel[:3]] + (pixel[3],) for pixel in pixels]
    new_image.putdata(new_pixels)

    # Save the new image
    new_image.save(output_path)

def main():
    image_path = "chainmail_layer_2.png"
    output_path = "copper_layer_2.png"

    identify_and_replace_colors(image_path, output_path)
    print(f"New image with randomized colors saved to {output_path}")

if __name__ == "__main__":
    main()
