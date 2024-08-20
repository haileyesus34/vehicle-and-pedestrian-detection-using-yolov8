import torch
import cv2
import numpy as np

# Load YOLOv3 model
class YOLOv3:
    def __init__(self, model_path):
        self.model = torch.hub.load('ultralytics/yolov3', 'yolov3', pretrained=True)
        self.model.eval()
        self.model.cuda()  # Move model to GPU if available

    def detect_objects(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))  # Change from HWC to CHW
        img = np.expand_dims(img, axis=0)    # Add batch dimension
        img = torch.tensor(img, dtype=torch.float32).cuda() / 255.0  # Normalize and move to GPU
        
        with torch.no_grad():
            results = self.model(img)
        
        return results

    def draw_boxes(self, img, results):
        # Process results
        pred = results.pred[0].cpu().numpy()  # Get predictions
        img_h, img_w = img.shape[:2]
        
        for det in pred:
            if len(det) > 0:
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    label = f'{self.model.names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img

def main():
    # Load model
    yolo = YOLOv3(model_path='yolov3')

    # Read image
    img = cv2.imread('path_to_your_image.jpg')
    
    # Detect objects
    results = yolo.detect_objects(img)
    
    # Draw bounding boxes
    img_with_boxes = yolo.draw_boxes(img, results)
    
    # Show image
    cv2.imshow('YOLOv3 Object Detection', img_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
