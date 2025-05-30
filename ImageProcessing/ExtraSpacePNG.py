import os
from PIL import Image, ImageDraw
import re
import torch


''' Given a directory of extracted bubbles, creates another directory of said bubbles pasted on pages 

1000 bubbles will produce 11 pages, first 10 pages have 96 bubbles each

Args:
    fileName: Specific directory for extracted bubbles, for iterating through model and eps value
    bubble_directory: Base directory for extracted bubbles
    misclassificationLabels: Boolean for extrated bubble title, set to True if contains __(Correct|Misclassified)
    saveDir: Directory to save pages

Note: Only toggle misclassificationLabels = True if bubble_directory was created using DisplayImgs() in Utilities.VoterLab_Classifier_Functions.py with wrongLocation provided
'''
misclassificationLabels = False

# Constants for sheet layout
dpi = 200
PAGE_WIDTH = int(8.5 * dpi)  # 8.5 inches converted to pixels at 200 dpi
PAGE_HEIGHT = int(11 * dpi)  # 11 inches converted to pixels at 200 dpi
MARGIN = int(1 * dpi)  # 1 inch margin converted to pixels at 200 dpi
SPACING = int(0.5 * dpi)  # 0.5 inch spacing converted to pixels at 200 dpi
BUBBLE_WIDTH = 50  # Bubble width in pixels
BUBBLE_HEIGHT = 40  # Bubble height in pixels

# Function to extract batch_index and example_index from the filename
def get_batch_and_example_indices(filename):
    if misclassificationLabels:
        pattern = r"(\d+)th Batch (\d+)th Example__(Correct|Misclassified)_(Vote|Non-Vote).png" 
    else: 
        pattern = r"__(\d+)th Batch (\d+)th Example__(Vote|Non-Vote).png" 
    match = re.match(pattern, filename)

    if match:
        batch_index = int(match.group(1))
        example_index = int(match.group(2))
        return batch_index, example_index
    else:
        raise ValueError(f"Invalid filename format: {filename}")

# Function to extract batch_index, example_index, and label from the filename
def get_batch_example_and_label(filename):
    if misclassificationLabels:
        pattern = r"(\d+)th Batch (\d+)th Example__(Correct|Misclassified)_(Vote|Non-Vote).png" 
    else: 
        pattern = r"__(\d+)th Batch (\d+)th Example__(Vote|Non-Vote).png" 
    match = re.match(pattern, filename)

    if match:
        batch_index = int(match.group(1))
        example_index = int(match.group(2))
        if misclassificationLabels: 
            classification = match.group(3)
            vote_type = match.group(4)
            return batch_index, example_index, classification, vote_type
        else: 
            vote_type = match.group(3)
            return batch_index, example_index, "None", vote_type
    else:
        # Return default values in case the label is not present in the filename
        return -1, -1, "Unknown"


def main(fileName, bubble_directory, misclassificationLabels, saveDir):
    # Get the list of PNG files in the bubble directory, sort them based on batch and example indices
    png_files = [file for file in os.listdir(bubble_directory) if file.endswith(".png")]
    png_files.sort(key=get_batch_and_example_indices)

    # Calculate the number of bubbles that can fit on each sheet
    available_width = PAGE_WIDTH - 2 * MARGIN
    available_height = PAGE_HEIGHT - 2 * MARGIN
    num_columns = available_width // (BUBBLE_WIDTH + SPACING)
    num_rows = available_height // (BUBBLE_HEIGHT + SPACING)
    total_bubbles_per_sheet = num_columns * num_rows

    # Initialize counters
    sheet_index = 1
    bubbles_placed = 0

    # Initialize an empty list to store the labels (0 for "Non-Vote", 1 for "Vote")
    labels = []

    while bubbles_placed < len(png_files):
        # Create a new blank sheet
        sheet = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
        draw = ImageDraw.Draw(sheet)

        # Place the bubbles on the sheet
        x = MARGIN
        y = MARGIN

        for index in range(bubbles_placed, min(bubbles_placed + total_bubbles_per_sheet, len(png_files))):
            if (index - bubbles_placed) % num_columns == 0 and index != bubbles_placed:
                # Move to the next row
                x = MARGIN
                y += BUBBLE_HEIGHT + SPACING

            bubble_file = png_files[index]
            bubble_path = os.path.join(bubble_directory, bubble_file)
            bubble = Image.open(bubble_path)

            sheet.paste(bubble, (x, y))

            # Get the label from the filename and add it to the labels list
            _, _, _, label = get_batch_example_and_label(bubble_file)
            label_value = 0 if label == "Vote" else 1
            labels.append(label_value)

            x += BUBBLE_WIDTH + SPACING

        # Save the sheet in the PreImagesSimpleCNN_Val directory
        # (("0" + str(sheet_index)) if (sheet_index < 10) else str(sheet_index))
        trailing_sheet_index = str(sheet_index).zfill(3)
        output_path = str(saveDir) + "/output_sheet_" + trailing_sheet_index + ".png" 
        #output_path = f"{saveDir}/output_sheet_{sheet_index}.png"
        sheet.save(output_path)

        # Update the counters
        sheet_index += 1
        bubbles_placed += total_bubbles_per_sheet

    print(f"{sheet_index - 1} sheets have been generated in the " + fileName + " directory.")
    #print("Labels array:", labels)
    print("Num labels:" + str(len(labels)))
    # Save labels in .torch file
    torch.save(labels, os.path.join(saveDir, fileName + '.torch'))


# Example below...
if __name__ == '__main__':
    fileName = 'Bubbles'
    bubble_directory = os.getcwd() + '/Bubble_Dir'
    saveDir = os.getcwd() + '/Bubble_Pages'
    os.makedirs(saveDir, exist_ok=True)
    main(fileName, bubble_directory, False, saveDir)
