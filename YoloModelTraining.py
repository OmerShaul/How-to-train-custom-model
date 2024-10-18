from ultralytics import YOLO

# Load a model
model = YOLO(r"C:\Users\hayim\Documents\Babypal\yolo11n.pt")  # build a new model from scratch

# Use the model
results = model.train(data=r"C:\Users\hayim\Documents\Babypal\Codes\Config.yaml", epochs=110)  # train the model with more epochs for bigger, slower, and more complex models
                      