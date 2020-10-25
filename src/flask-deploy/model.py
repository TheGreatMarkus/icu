import cv2
import numpy as np
import base64

locked = 0

def get_image_original(file_name):
    img = cv2.imread(file_name)
    return img


def base64_str_to_img(base64str):
    if (base64str.startswith("data:")):
        base64str = base64str.split(",")[1]
    img_bytes = base64.b64decode(base64str)
    im_arr = np.frombuffer(img_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img


def get_image(file_name):
    with open(file_name, "rb") as f:
        img_b64 = base64.b64encode(f.read())

    img_bytes = base64.b64decode(img_b64)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)
    return img


def predict_base64(img_str: str):
    img = base64_str_to_img(img_str)
    return predict_img(img)


def predict_img(img: np.ndarray):
    classes = []
    with open('../../files/coco.names', 'r') as f:
        classes = f.read().splitlines()

    net = cv2.dnn.readNet('../../files/yolov3.weights', '../../files/yolov3.cfg')

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)

    global locked
    locked = locked + 1
    print("START {}".format(locked))

    try:

        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)


        bounding_boxes = []
        confidences = []
        class_ids = []

        # first four elements in the output are bounding box coordinates
        # from 5th element on are scores
        for output in layerOutputs:
            for detection in output:
                # store all detections for different classes
                scores = detection[5:]
                # the classes that has is the most likely
                class_id = np.argmax(scores)
                # extract the max scores
                confidence = scores[class_id]

                if confidence > 0.5:
                    # center coordinate of the detected object
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)

                    # width and height of the bounding box
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # position of the upper corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    bounding_boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        # keep the most probable boxes because we can have more than 1 box for the same object (?)
        indexes = cv2.dnn.NMSBoxes(bounding_boxes, confidences, 0.5, 0.4)

        result = []

        # display bounding box on the image
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = bounding_boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))

                result.append({'label': label, 'confidence': confidence, 'bounds': bounding_boxes[i]})
    except:
        print("error END")
        locked = locked - 1
        return
    print(result)
    print("END")
    locked = locked - 1
    return result
