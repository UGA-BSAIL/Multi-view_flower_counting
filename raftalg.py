import sys
sys.path.append('core')
import argparse
import torch
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder

class Raftalg:
    def __init__(self, weight, device):
        parser = argparse.ArgumentParser()
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        args = parser.parse_args()
        self.model = torch.nn.DataParallel(RAFT(args))
        self.model.load_state_dict(torch.load(weight))
        self.model = self.model.module
        self.model.to(device)
        self.model.eval()

    def calculateopticflow(self, img1, img2):
        with torch.no_grad():
            padder = InputPadder(img1.shape)
            image1, image2 = padder.pad(img1, img2)
            flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)
        return flow_up




