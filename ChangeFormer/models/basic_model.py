import os

import torch

from misc.imutils import save_image
from models.networks import *


class CDEvaluator():

    def __init__(self, args):

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0]
                                   if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")

        print(self.device)

        self.checkpoint_dir = args.checkpoint_dir

        self.pred_dir = args.output_folder
        os.makedirs(self.pred_dir, exist_ok=True)

    def load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            # Handle PyTorch 2.6+ default weights_only=True by forcing weights_only=False when supported
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            except TypeError:
                # Older PyTorch versions do not support weights_only
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Support both full checkpoint dicts and raw state_dicts
            state_dict = checkpoint.get('model_G_state_dict') if isinstance(checkpoint, dict) else None
            if state_dict is None and isinstance(checkpoint, dict):
                # Some checkpoints may save the model weights directly under a common key
                for key in ['state_dict', 'model_state_dict', 'G_state_dict']:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        break
            if state_dict is None:
                # Assume the checkpoint itself is a state_dict
                state_dict = checkpoint

            self.net_G.load_state_dict(state_dict, strict=False)
            self.net_G.to(self.device)

            # Update optional metadata if present
            if isinstance(checkpoint, dict):
                self.best_val_acc = checkpoint.get('best_val_acc')
                self.best_epoch_id = checkpoint.get('best_epoch_id')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)
        return self.net_G


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.shape_h = img_in1.shape[-2]
        self.shape_w = img_in1.shape[-1]
        self.G_pred = self.net_G(img_in1, img_in2)[-1]
        return self._visualize_pred()

    def eval(self):
        self.net_G.eval()

    def _save_predictions(self):
        """
        保存模型输出结果，二分类图像
        """

        preds = self._visualize_pred()
        name = self.batch['name']
        for i, pred in enumerate(preds):
            file_name = os.path.join(
                self.pred_dir, name[i].replace('.jpg', '.png'))
            pred = pred[0].cpu().numpy()
            save_image(pred, file_name)

