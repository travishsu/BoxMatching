import numpy as np


def yolo_box(gt_box, image_width, image_height, cell_length):
    num_class = max([i[0] for i in gt_box]) + 1
    cell_width, cell_height = image_width // cell_length, image_height // cell_length

    target = np.zeros((cell_width, cell_height, 1 + 4 + num_class))
    for box in gt_box:
        class_num = box[0]
        l, t, r, b = box[1]
        box_middle_point_x, box_middle_point_y = .5 * (l + r), .5 * (t + b)
        x_cell, y_cell = box_middle_point_x // cell_length, box_middle_point_y // cell_length

        target[x_cell, y_cell, :5] = np.array([1, l, t, r, b])
        target[x_cell, y_cell, 5 + class_num] = 1

    return target.reshape(cell_width * cell_height, -1)
