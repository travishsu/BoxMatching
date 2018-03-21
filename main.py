import numpy as np

from preprocess import yolo_box

image_width, image_height = 1024, 768
cell_length = 256

# Box format: [class, [l, t, r, b]]
gt_box = [
    [0, [56, 212, 80, 300]],
    [3, [212, 56, 300, 80]]
]
pd_box = [
    [0, [57, 212, 81, 295]],
]

target = yolo_box(gt_box, 1024, 768, cell_length)
print(target)