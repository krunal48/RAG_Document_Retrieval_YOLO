from yolo import YoloExtractor
import os

print(os.path.dirname(os.path.abspath(__file__))+'\\'+'cache')
yoloE = YoloExtractor(os.path.dirname(os.path.abspath(__file__))+'\\'+'cache')
indexes = yoloE.extract(clear=False)

print(indexes)

