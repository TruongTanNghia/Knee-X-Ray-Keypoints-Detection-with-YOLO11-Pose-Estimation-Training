import cv2
from ultralytics import YOLO
import supervision as sv
import pyresearch

# Load YOLO model
model = YOLO("last.pt")

def process_image(input_image: str, output_image: str):
    # Read the image
    image = cv2.imread(input_image)
    if image is None:
        print("Error: Unable to read the image.")
        return

    # Resize the image
    resized = cv2.resize(image, (640, 640))

    # Perform detection
    detections = sv.Detections.from_ultralytics(model(resized)[0])

    # Annotate the image
    annotated = sv.BoundingBoxAnnotator().annotate(scene=resized, detections=detections)
    annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=detections)

    # Save the annotated image
    cv2.imwrite(output_image, annotated)
    print(f"Processed and saved: {output_image}")

if __name__ == "__main__":
    process_image("demo.jpg", "annotated_demo.jpg")
