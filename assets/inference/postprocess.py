import numpy as np

def intersection(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    width = np.clip(x2 - x1, a_min=0, a_max=None)
    height = np.clip(y2 - y1, a_min=0, a_max=None)
    
    intersection_area = width * height
    
    return intersection_area

def box_area(box):
    dx = box[:, 2] - box[:, 0]
    dy = box[:, 3] - box[:, 1]

    return dx * dy

def remove_overlap(box):
    area = box_area(box)
    indices = np.argsort(area)[::-1]
    
    threshold = 0.9
    keep = []
    
    for i in range(len(indices)):
        valid = True
        
        for j in range(i + 1, len(indices)):
            inter = intersection(box[indices[i]], box[indices[j]])
            if inter > threshold * min(area[indices[i]], area[indices[j]]):
                valid = False
                continue
        
        if valid:
            keep.append(indices[i])
            
    return np.array(keep, dtype=int)