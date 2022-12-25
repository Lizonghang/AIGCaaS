import os
import argparse
import warnings
import torch
import piq
import pickle
import blobfile as bf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

title_fontsize = 14
legend_fontsize = 13
label_fontsize = 14
tick_fontsize = 13
text_fontsize = 12
colors = ("m", "orange", "b", "gray", "g", "purple", "pink")
hatches = ("x", "/", "\\", "-")


def imread(path):
    with bf.BlobFile(path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
    pil_image = pil_image.convert("RGB")
    return pil_image


def reformat(x):
    np_x = np.array(x) / 255.
    np_x = np_x.swapaxes(0, 2)
    np_x = np.expand_dims(np_x, axis=0)
    return torch.Tensor(np_x)


def plot(result, metric="", intv=100, metric_func=None, save_as_file=True):
    met_name = "score" if metric_func else f"{metric} score"

    plt.figure(metric, figsize=(5, 4))
    plt.xlabel("num of transitions", fontsize=label_fontsize)
    plt.ylabel(met_name, fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    for im_id_, scores in result.items():

        xs = list(scores.keys())[::intv]

        if metric and metric_func is None:
            ys = [scores[x][metric] for x in xs]
        elif not metric and metric_func is not None:
            ys = [metric_func(**scores[x]) for x in xs]
        else:
            raise NotImplementedError("One of metric and metric_func must be set")

        plt.scatter(xs, ys, s=50, marker='o', c="b", alpha=0.5)

    plt.tight_layout()

    if save_as_file:
        save_dir = "./output"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"score_{metric}.png")
        plt.savefig(file_path)
        print(f"File saved at {file_path}")
    else:
        plt.show()


def custom_score(ssim=1., psnr=1., multi_scale_ssim=1., information_weighted_ssim=1.,
                 vif_p=1., fsim=1., srsim=1., gmsd=0., vsi=1., dss=1., haarpsi=1.,
                 mdsi=0., multi_scale_gmsd=0., total_variation=0., brisque=0.):
    score = ssim * psnr * multi_scale_ssim * information_weighted_ssim * \
            vif_p * fsim * srsim * vsi * dss * haarpsi * \
            (1 - gmsd) * (1 - mdsi) * (1 - multi_scale_gmsd) * \
            (1 - total_variation / 100.) * (1 - brisque / 100.)
    return max(score, 0.)


def calculate_scores(args):
    im_id, gt_path, inpainted_path, metrics_fr, metrics_nr = args
    res_ = {}

    # read origin image
    origin_image_path = os.path.join(gt_path, f"{im_id}.png")
    gt = imread(origin_image_path)
    gt = reformat(gt)
    assert gt.shape[1] == 3

    im_data_path = os.path.join(inpainted_path, im_id)

    # for the inpainted image in each step
    for file_ in tqdm(os.listdir(im_data_path), leave=True):
        n_step_ = int(file_.replace("step-", "").replace(".png", ""))
        res_[n_step_] = {}

        inpainted = imread(os.path.join(im_data_path, file_))
        inpainted = reformat(inpainted)
        assert gt.shape == inpainted.shape

        # calculate image quality
        for met_func in metrics_fr:
            met_name = met_func.__name__
            score = met_func(gt, inpainted)
            res_[n_step_][met_name] = score.item()

        for met_func in metrics_nr:
            met_name = met_func.__name__
            score = met_func(inpainted)
            res_[n_step_][met_name] = score.item()

    return im_id, res_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-path", type=str,
                        default="log/test_c256_nn2/gt")
    parser.add_argument("--inpainted-path", type=str,
                        default="log/test_c256_nn2/inpainted")
    parser.add_argument("--pkl-path", type=str,
                        default="log/test_c256_nn2/run_score.pkl")
    parser.add_argument("--load-pkl", type=bool, default=False)
    parser.add_argument("--num-proc", type=int, default=1)
    args = parser.parse_args()

    metrics_fr = [
        piq.ssim, piq.psnr, piq.ms_ssim.multi_scale_ssim,
        piq.iw_ssim.information_weighted_ssim, piq.vif_p,
        piq.fsim, piq.srsim, piq.gmsd, piq.vsi, piq.dss,
        piq.haarpsi, piq.mdsi, piq.multi_scale_gmsd]
    metrics_nr = [piq.total_variation, piq.brisque]

    image_ids = [im_name_[:6] for im_name_ in os.listdir(args.gt_path)]

    if not (args.load_pkl and os.path.exists(args.pkl_path)):
        if args.num_proc > 1:
            # multi-core processing
            mp_args = [(im_id_, args.gt_path, args.inpainted_path,
                        metrics_fr, metrics_nr) for im_id_ in image_ids]
            with Pool(args.num_proc) as p:
                result_multiprocess = p.map(calculate_scores, mp_args)
            p.close()
            p.join()

            # reformat result structure
            result = {}
            for im_id_, res_ in result_multiprocess:
                result[im_id_] = res_
        else:
            result = {}
            for im_id_ in image_ids:
                _, res_ = calculate_scores((
                    im_id_, args.gt_path, args.inpainted_path, metrics_fr, metrics_nr))
                result[im_id_] = res_

        # save as pickle file
        with open(args.pkl_path, "wb") as fp:
            pickle.dump(result, fp)

    # load from pickle file
    with open(args.pkl_path, "rb") as fp:
        result = pickle.load(fp)

    # visualize metrics
    vis_metrics = [
        "ssim", "psnr", "multi_scale_ssim", "information_weighted_ssim",
        "vif_p", "fsim", "srsim", "gmsd", "vsi", "dss", "haarpsi",
        "mdsi", "multi_scale_gmsd", "total_variation", "brisque"]

    for metric in vis_metrics:
        plot(result, metric=metric)
