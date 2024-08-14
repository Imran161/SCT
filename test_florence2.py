
# Commented out IPython magic to ensure Python compatibility.
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import copy
import supervision as sv
from PIL import Image
import torch

DEVICE = torch.device("cuda:2")

model_checkpoint = "/home/imran-nasyrov/model_checkpoints/1.1/epoch_120"
model_id = 'microsoft/Florence-2-large'
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, trust_remote_code=True).eval().to(DEVICE)
processor = AutoProcessor.from_pretrained(model_checkpoint, trust_remote_code=True)

"""## define the prediction function"""

def run_example(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
      input_ids=inputs["input_ids"].to(DEVICE),
      pixel_values=inputs["pixel_values"].to(DEVICE),
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer, image

"""## init image"""

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
# url = "https://www.culture.ru/s/slovo-dnya/peyzazh/images/tild3537-3033-4462-b061-313666313532__8.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

local_image_path = "/home/imran-nasyrov/sinusite_json_data/task_sinusite_data_29_11_23_1_st_sin_labeling/images/sinusite_29_11_23/sin_29_11_23_1/sinusite_29_11_23 (1145).jpg"
# local_image_path = "/home/imran-nasyrov/sct_project/sct_data/FINAL_CONVERT/100604476/1.2.392.200036.9116.2.6.1.48.1214245753.1506921568.925536/images/255_a8085fb96f3e8b87bd8723db0206120918052dcc566fb97d9f3de2f4fe789615_1.2.392.200036.9116.2.6.1.48.1214245753.1506921033.705311.png"
image = Image.open(local_image_path)

# task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
# task_prompt = "<DETAILED_CAPTION>" # {'CAPTION': '\nCT scan of the head and neck of a man with a large tumor in the middle of his head<loc_153><loc_113><loc_912><loc_998>\n'}
task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
results, modified_image = run_example(task_prompt, text_input="pathology")
print(results)

# OPEN_VOCABULARY_DETECTION короче тоже не рисует то что надо
def convert_to_od_format(data):  
    """  
    Converts a dictionary with 'bboxes' and 'bboxes_labels' into a dictionary with separate 'bboxes' and 'labels' keys.  
  
    Parameters:  
    - data: The input dictionary with 'bboxes', 'bboxes_labels', 'polygons', and 'polygons_labels' keys.  
  
    Returns:  
    - A dictionary with 'bboxes' and 'labels' keys formatted for object detection results.  
    """  
    # Extract bounding boxes and labels  
    bboxes = data.get('bboxes', [])  
    labels = data.get('bboxes_labels', [])  
      
    # Construct the output format  
    od_results = {  
        'bboxes': bboxes,  
        'labels': labels  
    }  
      
    return od_results 

# bbox_results  = convert_to_od_format(results['<OPEN_VOCABULARY_DETECTION>'])


import matplotlib.pyplot as plt  
import matplotlib.patches as patches  
def plot_bbox(image, data):
   # Create a figure and axes  
    fig, ax = plt.subplots()  

    # Display the image  
    ax.imshow(image)  

    # Plot each bounding box  
    for bbox, label in zip(data['bboxes'], data['labels']):  
        # Unpack the bounding box coordinates  
        x1, y1, x2, y2 = bbox  
        # Create a Rectangle patch  
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')  
        # Add the rectangle to the Axes  
        ax.add_patch(rect)  
        # Annotate the label  
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))  

    # Remove the axis ticks and labels  
    ax.axis('off')  

    # Show the plot  
    # plt.show()  
    save_path="test_florence"
    fig.savefig(f"{save_path}/img.jpg", bbox_inches='')

plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])    
# plot_bbox(image, bbox_results)   

    
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']
def draw_polygons(image, prediction, fill_mask=False, save_path=None):
    """
    Draws segmentation masks with polygons on an image.

    Parameters:
    - image_path: Path to the image file.
    - prediction: Dictionary containing 'polygons' and 'labels' keys.
                  'polygons' is a list of lists, each containing vertices of a polygon.
                  'labels' is a list of labels corresponding to each polygon.
    - fill_mask: Boolean indicating whether to fill the polygons with color.
    """
    # Load the image

    draw = ImageDraw.Draw(image)


    # Set up scale factor if needed (use 1 if not scaling)
    scale = 1

    # Iterate over polygons and labels
    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(colormap)
        fill_color = random.choice(colormap) if fill_mask else None

        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                print('Invalid polygon:', _polygon)
                continue

            _polygon = (_polygon * scale).reshape(-1).tolist()

            # Draw the polygon
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=fill_color)
            else:
                draw.polygon(_polygon, outline=color)

            # Draw the label text
            draw.text((_polygon[0] + 8, _polygon[1] + 2), label, fill=color)

    image.save(f"{save_path}/img.jpg")
    print(f"Saved image with polygons drawn to: {save_path}")

# output_image = copy.deepcopy(image)
# draw_polygons(output_image, results['<REFERRING_EXPRESSION_SEGMENTATION>'], fill_mask=True, 
#               save_path="test_florence")



