from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # initialize

result = model.train(data='config.yaml',  epochs=1, wandb=False)