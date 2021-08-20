"""Object detection using YOLOv5

Pokemon Pikachu detecting

"""

# import os, sys to append YOLOv5 folder path
import os, sys

# import object detection needed modules and libraries
# pillow
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch # PyTorch


# YOLOv5 folder path and related folder path settings
cwd = os.getcwd()
root_dir = (cwd + "/yolov5_stable")
sys.path.append(root_dir)


# import methods, functions from YOLOv5
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.plots import colors



# define a function to show detected pikachu
def show_pikachu(img, det):
    labels = ["pikachu"]
    img = Image.fromarray(img[...,::-1])
    draw = ImageDraw.Draw(img)
    font_size = max(round(max(img.size)/40), 12)
    font = ImageFont.truetype(cwd + "/yolov5_stable/fonts/times.ttf")

    for info in det:
        color = colors(1)
        target, prob = int(info[5].cpu().numpy()), np.round(info[4].cpu().numpy(), 2)
        x_min, y_min, x_max, y_max = info[0], info[1], info[2], info[3]
        draw.rectangle([x_min, y_min, x_max, y_max], width = 3, outline = color)
        draw.text((x_min, y_min), labels[target] + ':' + str(prob), fill = color, font = font)

    # Bug unresolved, pikachu shown in blue discolouration

    return img


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("GPU State: ", device)
    
    data_path = (cwd + "/test_data/")
    weight_path = (cwd + "/yolov5_stable/weights/best_v1.pt")
    dataset = LoadImages(data_path)
    model = attempt_load(weight_path, map_location = device)
    model.to(device)
    
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() # uint8 to fp16/32
        img /= 255.0 # 0-255 to 0.0-1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.25, 0.45)
        for i, det in enumerate(pred):
            im0 = im0s.copy()
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            result = show_pikachu(im0, det)
            result.show()



