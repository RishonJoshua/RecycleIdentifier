#!/usr/bin/python3
import jetson_inference
import jetson_utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--model", type=str, help="path to your custom .onnx model")
parser.add_argument("--labels", type=str, help="path to your labels.txt file")
parser.add_argument("--input_blob", type=str, default="input_0", help="input tensor name (check your onnx model)")
parser.add_argument("--output_blob", type=str, default="output_0", help="output tensor name (check your onnx model)")
opt = parser.parse_args()

img = jetson_utils.loadImage(opt.filename)

# Load your custom model directly
net = jetson_inference.imageNet(
    argv=[
        "--model=" + opt.model,
        "--labels=" + opt.labels,
        "--input_blob=" + opt.input_blob,
        "--output_blob=" + opt.output_blob
    ]
)

class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)
print("image is recognized as " + str(class_desc) + " (class #" + str(class_idx) + ") with " + str(confidence*100) + "% confidence")


###   IMPORTANT   ####
# Here, 40% as a minimal threshold is being utilized. The reason behind that is because the images that was available
# on kaggle was 100 pixels by 100 pixels. Therefore, some images do not have the best textures and patterns, things that 
# the resnet18 model utilizes to make a prediction. Some images also contain other items in the images further confusing the resnet18
# model. Do keep that in mind when interpretting results. That being said, there are some great categories that give good results. 

if str(class_desc) == 'apples' and int(confidence*100) >= 40:
    print("This item is compostable.")
elif str(class_desc) == 'pizza' and int(confidence*100) >= 40:
    print("This item is compostable.")
elif str(class_desc) == 'bananas' and int(confidence*100) >= 40:
    print("This item is compostable.")
elif str(class_desc) == 'phones' and int(confidence*100) >= 40:
    print("This item is recyclable.")
elif str(class_desc) == 'cups' and int(confidence*100) >= 40:
    print("This item is recyclable.")
elif str(class_desc) == 'bottles' and int(confidence*100) >= 40:
    print("This item is recyclable.")
else:
    print("This item is neither recyclable or compostable.")
