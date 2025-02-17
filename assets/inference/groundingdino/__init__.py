import torch
import os
from .checkpoints import download_checkpoint
from .util.slconfig import SLConfig
from .models import build_model
from .util.misc import clean_state_dict
import torchvision.transforms as T
from .util.utils import get_phrases_from_posmap
from torchvision.ops import box_convert, nms

GroundingDinoDir = os.path.dirname(os.path.realpath(__file__))

class GroundingDino:
    def __init__(self, config_path=None, checkpoint_path=None, device="cuda", tokenizer_path=None):
        if config_path is None:
            config_path = os.path.join(GroundingDinoDir, 'config', 'GroundingDINO_SwinT_OGC.py')

        if checkpoint_path is None:
            checkpoint_path = os.path.join(GroundingDinoDir, 'checkpoints', 'groundingdino_swint_ogc.pth')

            if not os.path.exists(checkpoint_path):
                download_checkpoint(checkpoint_path)

        self.device = device
        
        args = SLConfig.fromfile(config_path)
        args.device = self.device
        args.tokenizer_path = tokenizer_path
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model = build_model(args)
        
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        model = model.to(self.device)

        self.model = model

        self.transforms = T.Compose([
            T.Resize(size=800, max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict(self, image, caption, box_threshold, text_threshold):
        caption = caption.lower().strip()

        if not caption.endswith("."):
            caption = caption + "."
        
        image = image.to(self.device)
        outputs = self.model(image[None], captions=[caption])

        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

        mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits = prediction_logits[mask]  # logits.shape = (n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

        tokenizer = self.model.tokenizer
        tokenized = tokenizer(caption)

        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]

        return boxes, logits.max(dim=1)[0], phrases

    @torch.no_grad()
    def __call__(self, image, caption, box_threshold = 0.35, text_threshold = 0.25, nms_threshold=None):
        processed_image = self.transforms(image).to(self.device)

        boxes, logits, phrases = self.predict(processed_image, caption, box_threshold, text_threshold)

        if nms_threshold is not None and nms_threshold != 'None':
            keep = nms(boxes, logits, nms_threshold)
            boxes, logits, phrases = boxes[keep], logits[keep], [phrases[_] for _ in keep]

        h, w = image.height, image.width
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        logits = logits.numpy()

        return boxes, logits, phrases

