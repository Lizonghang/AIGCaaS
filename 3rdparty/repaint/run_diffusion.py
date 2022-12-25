# Copyright (c) 2022 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This repository was forked from https://github.com/openai/guided-diffusion, which is under the MIT license

"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import os
import argparse
import torch as th
import torch.nn.functional as F
import conf_mgt
import numpy as np
from tqdm import tqdm
from utils import yamlread
from guided_diffusion import dist_util

# Workaround
try:
    import ctypes

    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402


def toU8(sample):
    if sample is None:
        return sample

    def _process_handler(x):
        x = ((x + 1) * 127.5).clamp(0, 255).to(th.uint8)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
        x = x.detach().cpu().numpy()
        return x

    res = None
    if type(sample) is th.Tensor:
        res = _process_handler(sample)
    elif type(sample) is list:
        res = [_process_handler(x_) for x_ in sample]
    return res


def main(conf: conf_mgt.Default_Conf):
    print("Start", conf['name'])

    device = dist_util.dev(conf.get('device'))

    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
        )

        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")
    dset = 'eval'
    eval_name = conf.get_default_eval_name()
    num_image_to_process = 10
    only_final = True if conf.mode == 't_T' else False
    dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    for batch in tqdm(iter(dl)):
        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}

        model_kwargs["gt"] = batch['GT']

        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=device
            )
            model_kwargs["y"] = classes

        t_iter = tqdm(np.arange(10, 260, step=10), leave=False) if conf.mode == 't_T' \
            else [conf.schedule_jump_params['t_T']]

        for T_ in t_iter:
            if conf.mode == 't_T':
                conf.schedule_jump_params['t_T'] = T_.item()
                _, diffusion = create_model_and_diffusion(
                    **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
                )

            sample_fn = (
                diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
            )

            result = sample_fn(
                model_fn,
                (batch_size, 3, conf.image_size, conf.image_size),
                clip_denoised=conf.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=device,
                progress=show_progress,
                only_final=only_final,
                conf=conf
            )

            srs = toU8(result['sample'])
            gts = toU8(result['gt'])
            lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                       th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))
            gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))
            step = result.get('step', None)

            conf.eval_imswrite(
                srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks, step=step, t_T=T_,
                img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)

            # release stored image cache
            del result, srs, gts, lrs, gt_keep_masks
            if conf.mode == 't_T': del diffusion
            if conf.mode == 'step': del step

        num_image_to_process -= 1
        if num_image_to_process == 0:
            break  # process only 20 image

    print("sampling complete")


if __name__ == "__main__":
    """
    Use case:
    
    python run_diffusion.py --conf-path confs/test_c256_thin.yml --mode t_T
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf-path', type=str, required=False, default=None)
    parser.add_argument('--mode', type=str, default='t_T', choices=['t_T', 'step'])
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    conf_arg.update({'mode': args.get('mode')})
    main(conf_arg)
