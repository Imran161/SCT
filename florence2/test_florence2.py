
# Commented out IPython magic to ensure Python compatibility.
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import copy
import supervision as sv
from PIL import Image
import torch

DEVICE = torch.device("cuda:2")
# DEVICE = torch.device("cpu")

model_checkpoint = "/home/imran-nasyrov/model_checkpoints/1.8/epoch_114"
# model_checkpoint = "/home/imran-nasyrov/model_checkpoints/1.2/epoch_120"
# model_checkpoint = "microsoft/Florence-2-base-ft"

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
    # print("generated_ids", generated_ids)
    # print("generated_ids shape", generated_ids.shape)
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print("generated_text", generated_text)
    # print("image.width", image.width)
    # print("image.height", image.height)
    
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer, image

"""## init image"""

# # url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
# url = "https://www.culture.ru/s/slovo-dnya/peyzazh/images/tild3537-3033-4462-b061-313666313532__8.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# local_image_path = "/home/imran-nasyrov/cvat_jsonl/task_task_21_mart_23_lindenbraten_num2c-2023_03_21_15_15_33-coco 1.0/images/6mart23 (6530).jpg"
# local_image_path = "/home/imran-nasyrov/cvat_jsonl/task_перелом-2022_11_30_20_31_59-coco 1.0/images/16.jpg"
# local_image_path = "/home/imran-nasyrov/cvat_mini/task_task_17_jule_23_pathology_anatomy_sinus_num2-2023_09_12_22_50_06-coco 1.0/images/105480892.jpg"

# train lung
local_image_path = "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_1c-2024_02_26_15_44_35-coco 1.0/images/13_10_23_1 (1).jpg"
# val lung
# local_image_path = "/home/imran-nasyrov/cvat_phrase/task_task_13_oct_23_pat_fut_5с-2024_04_09_10_42_06-coco 1.0/images/13_10_23_5 (1).jpg"
image = Image.open(local_image_path)


# task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
# task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>" # {'CAPTION': '\nCT scan of the head and neck of a man with a large tumor in the middle of his head<loc_153><loc_113><loc_912><loc_998>\n'}
task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
text_input = "hydrothorax"
results, modified_image = run_example(task_prompt, text_input=text_input)
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
def plot_bbox(image, data, text_input):
   # Create a figure and axes  
    fig, ax = plt.subplots()  

    # Display the image  
    ax.imshow(image)  
    
    image_width, image_height = image.size

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

    plt.text(image_width - 10, image_height - 10, text_input, color='white', fontsize=12, ha='right', va='bottom',
             bbox=dict(facecolor='black', alpha=0.7))
    
    # Remove the axis ticks and labels  
    ax.axis('off')  

    # Show the plot  
    # plt.show()  
    save_path="test_florence"
    
    if text_input != "":
        fig.savefig(f"{save_path}/{text_input}.jpg", bbox_inches='')
    else:
        fig.savefig(f"{save_path}/img.jpg", bbox_inches='')


plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'], text_input)    
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

output_image = copy.deepcopy(image)
# draw_polygons(output_image, results['<REFERRING_EXPRESSION_SEGMENTATION>'], fill_mask=True, 
#               save_path="test_florence")



# image.width 1024
# image.height 1024

# generated_ids tensor([[    2,     0, 50590,  ..., 50500, 50521,     2]])
# generated_ids shape torch.Size([1, 1025])
# generated_text <loc_321><loc_169><loc_408><loc_97><loc_551><loc_143><loc_551><loc_143><loc_544><loc_227>
# <loc_544><loc_227><loc_551><loc_252><loc_551><loc_252><loc_544><loc_253><loc_544><loc_252><loc_544><loc_252>
# <loc_546><loc_253><loc_546><loc_253><loc_547><loc_253><loc_547><loc_252><loc_547><loc_252><loc_548><loc_252>
# <loc_548><loc_253><loc_548><loc_253><loc_549><loc_253><loc_549><loc_252><loc_549><loc_252><loc_550><loc_252>
# <loc_550><loc_253><loc_550><loc_253><loc_551><loc_253><loc_551><loc_252><loc_550><loc_251><loc_550><loc_251>
# <loc_549><loc_251><loc_549><loc_252><loc_548><loc_251><loc_548><loc_251><loc_547><loc_251><loc_547><loc_252>
# <loc_546><loc_252><loc_546><loc_251><loc_546><loc_251><loc_547><loc_250><loc_547><loc_250><loc_546><loc_250>
# <loc_546><loc_251><loc_545><loc_251><loc_545><loc_252><loc_545><loc_252><loc_544><loc_251><loc_544><loc_251>
# <loc_543><loc_251><loc_543><loc_252><loc_543><loc_252><loc_542><loc_252><loc_542><loc_251><loc_542><loc_251>
# <loc_543><loc_250><loc_543><loc_250><loc_544><loc_250><loc_544>

# result
# {'<REFERRING_EXPRESSION_SEGMENTATION>': {'polygons': [[[329.21600341796875, 173.56800842285156, 418.30401611328125, 
# 99.84000396728516, 564.7360229492188, 146.94400024414062, 564.7360229492188, 146.94400024414062, 557.5680541992188, 
# 232.9600067138672, 557.5680541992188, 232.9600067138672, 564.7360229492188, 258.55999755859375, 564.7360229492188, 
# 258.55999755859375, 557.5680541992188, 259.5840148925781, 557.5680541992188, 258.55999755859375, 557.5680541992188,
# 258.55999755859375, 559.6160278320312, 259.5840148925781, 559.6160278320312, 259.5840148925781, 560.6400146484375, 
# 259.5840148925781, 560.6400146484375, 258.55999755859375, 560.6400146484375, 258.55999755859375, 561.6640014648438, 
# 258.55999755859375, 561.6640014648438, 259.5840148925781, 561.6640014648438, 259.5840148925781, 562.6880493164062, 
# 259.5840148925781, 562.6880493164062, 258.55999755859375, 562.6880493164062, 258.55999755859375, 563.7120361328125, 
# 258.55999755859375, 563.7120361328125, 259.5840148925781, 563.7120361328125, 259.5840148925781, 564.7360229492188,
# 259.5840148925781, 564.7360229492188, 258.55999755859375, 563.7120361328125, 257.5360107421875, 563.7120361328125, 
# 257.5360107421875, 562.6880493164062, 257.5360107421875, 562.6880493164062, 258.55999755859375, 561.6640014648438, 
# 257.5360107421875, 561.6640014648438, 257.5360107421875, 560.6400146484375, 257.5360107421875, 560.6400146484375, 
# 258.55999755859375, 559.6160278320312, 258.55999755859375, 559.6160278320312, 257.5360107421875, 559.6160278320312, 
# 257.5360107421875, 560.6400146484375, 256.51202392578125, 560.6400146484375, 256.51202392578125, 559.6160278320312, 
# 256.51202392578125, 559.6160278320312, 257.5360107421875, 558.592041015625, 257.5360107421875, 558.592041015625, 
# 258.55999755859375, 558.592041015625, 258.55999755859375, 557.5680541992188, 257.5360107421875, 557.5680541992188, 