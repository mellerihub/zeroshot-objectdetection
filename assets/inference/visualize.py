import cv2
import numpy as np
import supervision as sv

def annotate(image, boxes, logits, phrases):
    detections = sv.Detections(xyxy=boxes)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = np.array(image)
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    return annotated_frame


def annotate_mask(image, masks, box_coords=None, sod=False):
    masks = (masks[:, 0, :, :] != 0)

    annotated_frame = np.array(image)

    if sod:
        masks = np.any(masks, axis=0)[:, :, None]
        return annotated_frame * masks

    if box_coords is None:
        detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=masks), mask=masks)

    else:
        detections = sv.Detections(xyxy=box_coords, mask=masks)

    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

    if box_coords is not None:
        bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)

    return annotated_frame