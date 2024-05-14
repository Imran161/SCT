'''
Чтобы превратить маскy NumPy в формат COCO (Common Objects in Context), вам необходимо выполнить несколько шагов. Формат COCO используется для аннотации изображений, 
в том числе для задач сегментации объектов, и включает в себя информацию о положении объектов, их категориях и сегментационных масках. Сегментационные маски в COCO 
представлены в виде полигонов или в формате RLE (Run-Length Encoding).

Вот основные шаги для преобразования маски NumPy в COCO дикт:

Преобразование маски NumPy в бинарный формат: убедитесь, что ваша маска представляет собой бинарный массив NumPy, где пиксели объекта имеют значение 1, а фон - 0.

Нахождение контуров объекта: используйте функцию findContours из библиотеки OpenCV для нахождения контуров объекта в бинарной маске.

Преобразование контуров в полигоны COCO: преобразуйте контуры, полученные на предыдущем шаге, в формат полигонов, подходящих для COCO. Это включает в себя конвертацию 
координат контуров в список точек полигонов.

Преобразование в RLE или сохранение в виде полигонов: в зависимости от того, какой формат вам нужен, вы можете либо оставить полигоны как есть, либо преобразовать их в формат 
RLE с помощью специфических для COCO инструментов или библиотек.

Создание COCO аннотации: создайте словарь с информацией об аннотации, следуя структуре COCO. Это включает в себя, помимо прочего, поля segmentation (для полигонов или RLE), 
area (площадь объекта), bbox (ограничивающий прямоугольник) и category_id (идентификатор категории объекта).

Приведу пример кода на Python, демонстрирующий основные этапы этого процесса:

В этом примере используется библиотека pycocotools для работы с RLE и расчета площади и ограничивающего прямоугольника. Убедитесь, что у вас установлены все необходимые 
библиотеки (numpy, cv2 и pycocotools), и что ваша начальная маска имеет правиль

'''


import numpy as np
import cv2
from pycocotools import mask as maskUtils


def numpy_mask_to_coco_polygon(numpy_mask):
    contours, _ = cv2.findContours(numpy_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [contour.flatten().tolist() for contour in contours]
    return polygons



def create_coco_annotation_from_mask(numpy_mask, category_id, image_id):
    polygons_first = numpy_mask_to_coco_polygon(numpy_mask) # polygons слева от = было
    
    # попробую так сделать 
    # Удаляем подсписки, длина которых меньше 6
    # print("polygons_first", len(polygons_first))
    filtered_list = [sublist for sublist in polygons_first if len(sublist) >= 6]
    
    if filtered_list == []:
        filtered_list = [[0, 0, 0, 0, 0, 0]] # надо понять что сюда писать 

    # похоже это кайфово работает
    polygons = filtered_list
    # print("polygons", polygons)
    # print("len polygons)", len(polygons))
    # тут может быть несвязнвя область полигона и тогда ошибка хотя раньше все работало

    # if polygons_first == [[]]:
    #     rle = [{'size': [512, 512], 'counts': b'PPP8'}]
    #     area = [0]
    #     bbox = [[0, 0, 0, 0]]
    # else:
    rle = maskUtils.frPyObjects(polygons, numpy_mask.shape[0], numpy_mask.shape[1])
    area = maskUtils.area(rle)
    area = area.tolist()
    bbox = maskUtils.toBbox(rle)
    bbox = bbox.tolist()
    # print("rle", rle)
    # print("area", area)
    # print("bbox", bbox)
    
    # было так для моих данных
    ###############################################
    # parts = image_id.split('_')
    # id_number = parts[0]
    # id_number = int(id_number)
    # print("image_id", image_id)
    # print("id_number", id_number)
    ###############################################
    
    # для FINAL_CONVERT_NEW вот так 
    # parts = image_id.split('.')
    # id_number = parts[0]
    # id_number = int(id_number)
    # # print("image_id", image_id)
    # # print("id_number", id_number)
    #############################
    
    
    # для айнура так
    ###############################################

    # print("image_id", image_id) # image_id level_0_117.jpg
    parts = image_id.split('_')
    id_number = parts[-1].split(".")[0]
    id_number = int(id_number)
    # print("id_number in coco", id_number) # id_number in coco 117
    ###############################################
    
    
    
    annotation = {
        "segmentation": polygons, 
        "area": area, #.tolist(),
        "bbox": bbox, #.tolist(),
        "category_id": int(category_id),
        "image_id": id_number,
        "iscrowd": 0,
        "id": id_number, # ?
        "attributes": {
            "occluded": bool(0)
        }
    }
    
    images = {
        "license": 0,
        "file_name": image_id, # image_id, для моих данных вот так
        "coco_url": "",
        "height": 256, # для айнура 256
        "width": 256, # и тут 
        "date_captured": "",
        "flickr_url": "",
        "id": id_number # ? тоже саме что и image_id в annotation ?
    }
    
    return images, annotation# , categories


coco_dataset = {
    "info": {        
        "version": "",
        "date_created": "",
        "contributor": "",
        "year": 2024, # было ""
        "description": "",
        "url": ""
    },
    
    "licenses": [    
        {
            "url": "",
            "id": 0,
            "name": ""
        }
    ],
    
    "images": [],
    "annotations": [],
    # "categories": []
    "categories": [
        {
            "id": 0,
            "name": "0", 
            "supercategory": ""
        },
        {
            "id": 1,
            "name": "1",
            "supercategory": ""
        },
        {
            "id": 2,
            "name": "2",
            "supercategory": ""
        },
        {
            "id": 3,
            "name": "3",
            "supercategory": ""
        },
        {
            "id": 4,
            "name": "4",
            "supercategory": ""
        },
        {
            "id": 5,
            "name": "5",
            "supercategory": ""
        },
        {
            "id": 6,
            "name": "6",
            "supercategory": ""
        },
        {
            "id": 7,
            "name": "7",
            "supercategory": ""
        },
        {
            "id": 8,
            "name": "8",
            "supercategory": ""
        },
        {
            "id": 9,
            "name": "9",
            "supercategory": ""
        },
        {
            "id": 10,
            "name": "10",
            "supercategory": ""
        },
        {
            "id": 11,
            "name": "11",
            "supercategory": ""
        },
        {
            "id": 12,
            "name": "12",
            "supercategory": ""
        },
        {
            "id": 13,
            "name": "13",
            "supercategory": ""
        },
        {
            "id": 14,
            "name": "14",
            "supercategory": ""
        },
        {
            "id": 15,
            "name": "15",
            "supercategory": ""
        },
        {
            "id": 16,
            "name": "16",
            "supercategory": ""
        },
        {
            "id": 17,
            "name": "17",
            "supercategory": ""
        },
        {
            "id": 18,
            "name": "18",
            "supercategory": ""
        },
        {
            "id": 19,
            "name": "19",
            "supercategory": ""
        },
        {
            "id": 20,
            "name": "20",
            "supercategory": ""
        },
        {
            "id": 21,
            "name": "21",
            "supercategory": ""
        },
        {
            "id": 22,
            "name": "22",
            "supercategory": ""
        },
        {
            "id": 23,
            "name": "23",
            "supercategory": ""
        },
        {
            "id": 24,
            "name": "24",
            "supercategory": ""
        }

    ]
}



# почему-то не могу заимпортить ее
def draw_image_from_polygon(polygon, image_shape):
    image = np.zeros(image_shape, dtype=np.uint8)
    polygon = np.array(polygon).reshape((-1, 2)).astype(np.int32)
    cv2.fillPoly(image, [polygon], (255, 255, 255))
    return image



import numpy as np
import cv2
from pycocotools import mask as maskUtils


def numpy_mask_to_coco_polygon(numpy_mask):
    contours, _ = cv2.findContours(numpy_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [contour.flatten().tolist() for contour in contours]
    return polygons



def create_empty_coco_annotation_from_mask(image_id):
    
    parts = image_id.split('.')
    id_number = parts[0]
    id_number = int(id_number)
    # print("image_id", image_id)
    # print("id_number", id_number)


    
    images = {
        "license": 0,
        "file_name": image_id, # image_id, для моих данных вот так
        "coco_url": "",
        "height": 256,
        "width": 256,
        "date_captured": "",
        "flickr_url": "",
        "id": id_number # ? тоже саме что и image_id в annotation ?
    }
    
    return images
