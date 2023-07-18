import torch


def intersection_over_union(y_true, y_pred, format="midpoint"):
    if format == "midpoint":
        box1_x1 = y_true[..., 0:1] - y_true[..., 2:3] / 2
        box1_x2 = y_true[..., 0:1] + y_true[..., 2:3] / 2
        box1_y1 = y_true[..., 1:2] - y_true[..., 3:4] / 2
        box1_y2 = y_true[..., 1:2] + y_true[..., 3:4] / 2

        box2_x1 = y_pred[..., 0:1] - y_pred[..., 2:3] / 2
        box2_x2 = y_pred[..., 0:1] + y_pred[..., 2:3] / 2
        box2_y1 = y_pred[..., 1:2] - y_pred[..., 3:4] / 2
        box2_y2 = y_pred[..., 1:2] + y_pred[..., 3:4] / 2
    else:
        raise ValueError("No format provided")

    x1 = torch.max(box1_x1, box2_x1)[0]
    y1 = torch.max(box1_y1, box2_y1)[0]
    x2 = torch.min(box1_x2, box2_x2)[0]
    y2 = torch.min(box1_y2, box2_y2)[0]

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1 = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2 = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    union = box1 + box2 - inter + 1e-6

    iou = inter / union

    return iou


def nms(boxes, iou_threshold=0.5, threshold=0.4, format="midpoint"):
    boxes = [box for box in boxes if box[1] > threshold]
    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
    box_after_nms = []

    while boxes:
        max_box = boxes.pop(0)
        boxes = [box for box in boxes if
                 box[0] != max_box[0] or intersection_over_union(torch.tensor(max_box[2:]), torch.tensor(box[2:])) > iou_threshold]
        box_after_nms.append(max_box)

    return box_after_nms
