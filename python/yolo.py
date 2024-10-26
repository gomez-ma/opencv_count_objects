import cv2
import time
import sys
import numpy as np
import os

def build_model(is_cuda):
    net = cv2.dnn.readNet("config_files/yolov5s.onnx")
    if is_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("Attempting to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4

def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds

def load_capture(source=0):
    if isinstance(source, str) and os.path.isfile(source):
        capture = cv2.VideoCapture(source)
    else:
        capture = cv2.VideoCapture(0)
    return capture

def load_classes():
    try:
        with open("config_files/classes.txt", "r") as f:
            class_list = [cname.strip() for cname in f.readlines()]
    except FileNotFoundError:
        print("Error: 'classes.txt' not found in 'config_files' directory.")
        class_list = []
    return class_list

class_list = load_classes()

# Find the class ID for "person" in the class list
person_class_id = class_list.index("person") if "person" in class_list else -1

def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]
    image_width, image_height, _ = input_image.shape
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if classes_scores[class_id] > SCORE_THRESHOLD:
                confidences.append(confidence)
                class_ids.append(class_id)
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = [left, top, width, height]
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)
    if len(indexes) > 0:
        indexes = indexes.flatten()
    else:
        indexes = []


    result_class_ids = [class_ids[i] for i in indexes]
    result_confidences = [confidences[i] for i in indexes]
    result_boxes = [boxes[i] for i in indexes]

    return result_class_ids, result_confidences, result_boxes

def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

def main(video_source=""):
    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
    is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"
    net = build_model(is_cuda)
    capture = load_capture(video_source)

    start = time.time_ns()
    frame_count = 0
    total_frames = 0
    fps = -1

    while True:
        ret, frame = capture.read()
        if not ret:
            print("End of stream")
            break

        inputImage = format_yolov5(frame)
        outs = detect(inputImage, net)

        class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

        # Count number of "person" detections
        person_count = sum(1 for class_id in class_ids if class_id == person_class_id)

        frame_count += 1
        total_frames += 1

        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        if frame_count >= 30:
            end = time.time_ns()
            fps = 1000000000 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()

        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the person count on the frame
        person_label = f"Person Count: {person_count}"
        cv2.putText(frame, person_label, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("output", frame)

        if cv2.waitKey(1) > -1:
            print("Finished by user")
            break

    print("Total frames: " + str(total_frames))
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()