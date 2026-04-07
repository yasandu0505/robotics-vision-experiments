import cv2
from ultralytics import YOLO

model = YOLO("yolov8s")

# initialize the webcam (0 is refers to the default camera)
cap = cv2.VideoCapture(0)

# start the infinite loop to continuously read frames from the webcam
while True:
    # read a frame from the webcam (ret == boolean indicating whether the frame was read successfully)
    # frame is the actual frame read from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, stream=True)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,0), 2)
    
    cv2.imshow("test 1", frame) # display the captured frame in a window named "test 1"

    if cv2.waitKey(1) & 0xFF == 27:
        break

# kill the webcam and close all windows
cap.release()
cv2.destroyAllWindows()