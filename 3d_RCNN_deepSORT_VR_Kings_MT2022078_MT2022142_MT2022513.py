from absl import flags
import sys

FLAGS = flags.FLAGS
FLAGS(sys.argv)

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import torchvision
import time


from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

transform = transforms.Compose([transforms.ToTensor(),])


def predict(image, model, device, detection_threshold):
    image = transform(image).to(device)
    image = image.unsqueeze(0)
    outputs = model(image)

    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    score, boxes = pred_scores[pred_scores >= detection_threshold],pred_bboxes[pred_scores >= detection_threshold].astype(np.float32)
    return boxes, pred_classes, outputs[0]['labels'], score

min_size = 800
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,min_size=min_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.eval().to(device)


vid = cv2.VideoCapture('./data/video/Video.mp4')

codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
vid_width, vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/results_video_rcnn.avi', codec, vid_fps, (vid_width, vid_height))

from _collections import deque

pts = [deque(maxlen=30) for _ in range(1000)]

counter = []

while True:
    _, img = vid.read()
    if img is None:
        print('Completed')
        break

    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    t1 = time.time()

    with torch.no_grad():
        boxes, classes, labels, scores = predict(img_in, model, device, 0.8)
    for box in boxes:
        box[0] = float(box[0] / img.shape[1])
        box[1] = float(box[1] / img.shape[0])
        box[2] = float(box[2] / img.shape[1])
        box[3] = float(box[3] / img.shape[0])

    converted_boxes = convert_boxes(img_in, boxes)
    features = encoder(img_in, boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(boxes, scores, classes, features)]

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    current_count = int(0)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        class_name = track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        if class_name=='car':
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)), (int(bbox[0]) + (len(class_name)
                                                                                   + len(str(track.track_id))) * 17,
                                                                   int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)

        center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
        pts[track.track_id].append(center)

        height, width, _ = img.shape
        cv2.line(img, (0, int(height/2+height/3)), (width, int(height/2 + height/3)), (0, 255, 0),
                 thickness=2)
        cv2.line(img, (0, int( height / 2 + height / 10)), (width, int( height / 2 + height / 10)), (0, 255, 0),
                 thickness=2)

        center_y = int(((bbox[1]) + (bbox[3])) / 2)

        if center_y <= int(height/2+height/3) and center_y >= int(height/2+height/10):
            if class_name == 'car':
                counter.append(int(track.track_id))
                current_count += 1

    total_count = len(set(counter))
    cv2.putText(img, "Current Vehicle Count: " + str(current_count), (0, 80), 0, 1, (0, 255, 0), 2)
    cv2.putText(img, "Total Vehicle Count: " + str(total_count), (0, 130), 0, 1, (0, 255, 0), 2)

    fps = 1. / (time.time() - t1)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30), 0, 1, (0, 0, 255), 2)
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output', 1024, 768)
    cv2.imshow('output', img)
    out.write(img)

    if cv2.waitKey(1) == ord('q'):
        break
vid.release()
out.release()
cv2.destroyAllWindows()
