from ultralytics import YOLO

# Load model
model = YOLO('models/best.pt')# yolov8 = 68.2 M parameters

# Inference
results=model.predict('input_videos/08fd33_4.mp4',save=True)

# First frame
print(results[0])

""""
 Each Bounding Box would be represented as a dictionary with the following keys:
 1. x,y,w,h:     The coordinates of the center of the bounding box and its width and height.
 2. x1,y1,x2,y2: The coordinates of the top-left and bottom-right corners of the bounding box. (Also called (x,y,x,y) format)

""" 
# Looping over the boxes in the first frame
for box in results[0].boxes:
    print(box)