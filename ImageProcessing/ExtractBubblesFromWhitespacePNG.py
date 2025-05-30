import os
from PIL import Image
import torch


''' Given a directory of registed pages, extract bubbles

Note: Use align_and_save_image if scanned images are .png format, else use align_tiff_and_save_image

Args:
    output_sheet_path: Path for registered page
    num_pages: Number of pages in output_sheet_path directory
    Labels_array: List of labels corresponding to each extracted bubble... found as a .torch file in save_dir from ExtraSpacePNG.py
    extracted_bubbles_directory: Directory for saving extracted bubbles
    numBubbles: Number of bubbles to extract
'''


def ExtractBubbles(output_sheet_path, num_pages, Labels_array, extracted_bubbles_directory, numBubbles):
    labelIndex = 0
    for k in range(1, num_pages + 1):
        # Open the output sheet image
        #output_sheet_path = f"/workspace/caleb/VoterLab/ImageProcessing/Positive_Gradient_PostPrint_Registered/TWINS_Clean/TWINS_Clean_{k}.png"
        output_sheet_page = output_sheet_path + "/output_sheet_" + ("00" if k < 10 else "0") + str(k) + ".png"
        print(output_sheet_page)
        #i = (k-1) * 24
        output_sheet = Image.open(output_sheet_page)

        # Constants for bubble layout
        dpi = 200
        BATCH_SIZE = 64
        PAGE_WIDTH = int(8.5 * dpi)  # 8.5 inches converted to pixels at 200 dpi
        PAGE_HEIGHT = int(11 * dpi)  # 11 inches converted to pixels at 200 dpi
        MARGIN = int(1 * dpi)  # 1 inch margin converted to pixels at 200 dpi
        BUBBLE_WIDTH = 50  # Bubble width in pixels
        BUBBLE_HEIGHT = 40  # Bubble height in pixels
        SPACING = int(0.5 * dpi)  # 0.5 inch spacing converted to pixels at 200 dpi

        # Calculate the number of bubbles that can fit on the sheet
        num_columns = int((PAGE_WIDTH - 2 * MARGIN) / (BUBBLE_WIDTH + SPACING))
        num_rows = int((PAGE_HEIGHT - 2 * MARGIN) / (BUBBLE_HEIGHT + SPACING))
        total_bubbles = num_columns * num_rows
        print("Total Number of Bubbles: " + str(total_bubbles))

        # Directory to save the extracted bubbles
        #extracted_bubbles_directory = "/workspace/caleb/VoterLab/ImageProcessing/Positive_Gradient_Adv_Bubbles/TWINS_CLEAN"

        # Create the directory if it doesn't exist
        os.makedirs(extracted_bubbles_directory, exist_ok=True)

        # Get the number of existing bubbles in the directory
        existing_bubbles = len(os.listdir(extracted_bubbles_directory))

        # Extract the individual bubbles from the sheet
        x = MARGIN
        y = MARGIN

        for row in range(num_rows):
            for col in range(num_columns):
                bubble_box = (x, y, x + BUBBLE_WIDTH, y + BUBBLE_HEIGHT)
                bubble = output_sheet.crop(bubble_box)

                # Save the extracted bubble
                if labelIndex < len(Labels_array):
                    Label = Labels_array[labelIndex]
                    batch_index = (existing_bubbles + row * num_columns + col) // BATCH_SIZE
                    example_index = (existing_bubbles + row * num_columns + col) % BATCH_SIZE
                    bubble_filename = f"{batch_index}th Batch {example_index}th Example_{Label}.png"
                    #print(bubble_filename)
                    bubble_path = os.path.join(extracted_bubbles_directory, bubble_filename)
                    bubble.save(bubble_path)

                x += BUBBLE_WIDTH + SPACING
                #print(i)
                labelIndex+=1

            # Move to the next row
            x = MARGIN
            y += BUBBLE_HEIGHT + SPACING

        print(f"Individual bubbles have been extracted and saved in the {output_sheet_path} directory.")


# Example below...
if __name__ == '__main__':
    output_dir = os.getcwd() + '/Bubble_Pages_Registered'
    os.makedirs(output_dir, exist_ok=True)
    num_pages = 10 
    Labels_array = torch.load(os.getcwd() + '/Bubble_Pages/Bubble.torch')
    extracted_bubbles_directory = os.getcwd() + '/Bubbles_Extracted'
    os.makedirs(output_dir, exist_ok=True)
    numBubbles = 96 * 10

    ExtractBubbles(
        output_sheet_path = output_dir
        num_pages = num_pages,
        Labels_array = Labels_array,
        extracted_bubbles_directory = extracted_bubbles_directory,
        numBubbles = numBubbles
    )
