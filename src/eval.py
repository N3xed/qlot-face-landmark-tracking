import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=".*albumentations.*")

from typing import Literal
from utils.torch.datasets import QueriedFaceDataset
from utils.torch.viz import draw_keypoints
from utils.datasets import WFLW, FaceSynthetics, IbugTest, DatasetName, ImageDataset
from utils.torch.misc import calc_nme, NMENormType
from model import QLOT, QueryPoints, LandmarkPrediction
from model.utils import NUM_PREDS_COORDS, NUM_PREDS_COV_PARAMS
from data import Datasets
from dataclasses import dataclass, field
import torch
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from numpy.typing import NDArray
from utils.torch.misc import load
import shutil

EvalImagesMode = Literal["worst_iod", "worst_size", "random", "best_iod", "best_size"] | list[int]


@dataclass(init=False)
class ImgEvalResult:
    img_paths: list[str]

    labels: torch.Tensor  # (num_images, num_queries, 2)
    xy: torch.Tensor  # (num_images, num_queries, 2)
    cov: torch.Tensor  # (num_images, num_queries, 3)
    nme_iod: torch.Tensor  # (num_images, num_queries)
    nme_s: torch.Tensor  # (num_images, num_queries)

    dataset_name: str
    _datasets: Datasets | None = field(init=False, repr=False)
    dataset: QueriedFaceDataset | None = field(init=False, repr=False)

    def __init__(self, datasets: Datasets, testset_name: str):
        self._datasets = datasets
        self.dataset_name = testset_name
        self.dataset = getattr(datasets, testset_name)
        assert self.dataset is not None

        self.num_images = len(self.dataset)
        self.nlandmarks = self.dataset.dataset.nlandmarks

        self.img_paths = []
        self.labels = torch.zeros((self.num_images, self.nlandmarks, NUM_PREDS_COORDS))
        self.xy = torch.zeros((self.num_images, self.nlandmarks, NUM_PREDS_COORDS))
        self.cov = torch.zeros((self.num_images, self.nlandmarks, NUM_PREDS_COV_PARAMS))
        self.nme_iod = torch.zeros((self.num_images, self.nlandmarks))
        self.nme_s = torch.zeros((self.num_images, self.nlandmarks))

    def summary(self):
        print(f"---- {self.dataset_name} ----")
        nme_iod = self.nme_iod.mean(dim=-1).cpu().numpy() * 100.0  # (num_images,)
        nme_s = self.nme_s.mean(dim=-1).cpu().numpy() * 100.0  # (num_images,)

        worst_nmes = nme_iod[np.argsort(-nme_iod)[:10]]
        print(f"Worst NMEs = [{', '.join([f'{nme:.2f}%' for nme in worst_nmes])}]")
        print(f"NME IOD = {nme_iod.mean():.2f}%")
        print(f"NME Size = {nme_s.mean():.2f}%")

        failure_rate_10 = (nme_iod > 10.0).sum() / len(nme_iod) * 100.0
        print(f"Failure Rate (NME > 10%) = {failure_rate_10:.2f}%")

    def ced_values(self, threshold: float = 10.0, norm_type: NMENormType = "iod") -> tuple[NDArray, NDArray]:
        STEP = 0.1
        thr = np.arange(0.0, threshold + STEP, STEP)
        ced_vals = []
        nme = self.nme_iod if norm_type == "iod" else self.nme_s
        nme = nme.mean(dim=-1).cpu().numpy() * 100.0  # (num_images,)
        for t in thr:
            ced = (nme <= t).sum() / len(nme) * 100.0
            ced_vals.append(ced)
        return thr, np.array(ced_vals)

    def display(
        self,
        title: str | None = None,
        figsize=(6, 6),
        row_cols=(1, 3),
        indices: EvalImagesMode = "random",
        resolution: int | None = 800,
        radius: int = 3,
        width: int = 2,
        print_indices: bool = False,
    ) -> plt.Figure:
        assert self.dataset is not None
        figw, figh = figsize
        rows, cols = row_cols
        figw *= cols
        figh *= rows

        fig, ax = plt.subplots(rows, cols, figsize=(figw, figh))
        axes: list[plt.Axes] = ax.flatten().tolist() if isinstance(ax, np.ndarray) else [ax]

        nme_iod = self.nme_iod.mean(dim=-1).cpu().numpy() * 100.0  # (num_images,)
        nme_s = self.nme_s.mean(dim=-1).cpu().numpy() * 100.0  # (num_images,)

        if isinstance(indices, str):
            if indices == "random":
                idx = np.random.randint(0, len(self.dataset), size=(len(axes),))
            elif indices == "worst_iod":
                idx = np.argsort(-nme_iod)[: len(axes)]
            elif indices == "worst_size":
                idx = np.argsort(-nme_s)[: len(axes)]
            elif indices == "best_iod":
                idx = np.argsort(nme_iod)[: len(axes)]
            elif indices == "best_size":
                idx = np.argsort(nme_s)[: len(axes)]
            else:
                raise ValueError(f"Invalid indices mode: {indices}")
        else:
            idx = np.array(indices)
        if print_indices:
            print(f"Displaying images at indices: {idx.tolist()}")

        gt, pd = None, None
        for i, ax in zip(idx, axes):
            sample = self.dataset[i]
            img = sample["image"].cpu()  # (C, H, W)
            all_xy = [
                self.labels[i].cpu(),  # Ground Truth,
                self.xy[i].cpu(),  # Predicted,
            ]
            try:
                from model import LowRankCov2D

                cov = LowRankCov2D(self.cov[i]).as_cov2d_params().cpu()  # (num_queries, 3)
                all_cov = [
                    torch.zeros_like(cov),  # Ground Truth (not available)
                    cov,  # Predicted
                ]
            except:
                all_cov = None
            img = draw_keypoints(
                img,
                all_xy,
                all_cov,
                colors=["green", "red"],
                probability_threshold=0.95,
                scale_to=resolution,
                radius=radius,
                width=width,
            )
            ax.imshow(img.permute(1, 2, 0).contiguous().numpy())
            ax.set_title(f"{self.img_paths[i]}")
            ax.axis("off")
            gt = ax.scatter([], [], color="green", label="Ground Truth")
            pd = ax.scatter([], [], color="red", label="Predicted")
            nme_iod_elem = ax.plot([], [], " ", label=f"NME IOD: {nme_iod[i]:.2f}%")
            nme_s_elem = ax.plot([], [], " ", label=f"NME Size: {nme_s[i]:.2f}%")
            ax.legend(handles=[*nme_iod_elem, *nme_s_elem], loc="upper right", handletextpad=0.0, handlelength=0)

        fig.legend(handles=[gt, pd], loc="upper right")
        if title is not None:
            fig.suptitle(title, fontsize=18)
            fig.tight_layout(rect=[0, 0, 1.0, 0.98], h_pad=2.0)  # type: ignore
        else:
            fig.tight_layout(h_pad=2.0)

        return fig

    def as_dict(self) -> dict:
        return {
            "num_images": self.num_images,
            "nlandmarks": self.nlandmarks,
            "img_paths": self.img_paths,
            "labels": self.labels.cpu().numpy(),
            "xy": self.xy.cpu().numpy(),
            "cov": self.cov.cpu().numpy(),
            "nme_iod": self.nme_iod.cpu().numpy(),
            "nme_s": self.nme_s.cpu().numpy(),
            "dataset_name": self.dataset_name,
        }

    @classmethod
    def from_dict(cls, data: dict, datasets: Datasets) -> "ImgEvalResult":
        dataset_name = data["dataset_name"]
        result = cls(datasets, dataset_name)
        result.img_paths = data["img_paths"]
        result.labels[...] = torch.from_numpy(data["labels"])
        result.xy[...] = torch.from_numpy(data["xy"])
        result.cov[...] = torch.from_numpy(data["cov"])
        result.nme_iod[...] = torch.from_numpy(data["nme_iod"])
        result.nme_s[...] = torch.from_numpy(data["nme_s"])
        return result



@torch.no_grad()
def eval_images(
    model: QLOT,
    model_queries: QueryPoints,
    datasets: Datasets,
    iod_indices: tuple[int, int],
    dataset_name: DatasetName,
    device: torch.device,
    iterations: int = 3,
    batch_size=64,
    split: str = "",
) -> ImgEvalResult:
    from tqdm import tqdm

    result = ImgEvalResult(datasets, dataset_name.get_testset_name(split))
    assert result.dataset is not None
    dataset = result.dataset
    assert isinstance(dataset.dataset, ImageDataset), f"Dataset {dataset_name} is not an ImageDataset"

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    queries = model_queries.get(dataset_name).to(device).unsqueeze(0)
    model.eval()

    nlandmarks = dataset.dataset.nlandmarks
    assert nlandmarks == queries.shape[1]

    for i, batch in enumerate(tqdm(dataloader)):
        images = batch["image"].to(device)
        curr_batch = images.shape[0]
        curr_queries = queries.expand(curr_batch, -1, -1)

        preds: LandmarkPrediction
        preds = model(images, curr_queries, iterations=iterations)

        xy = preds.mean  # (batch_size, num_queries, 2)
        labels: torch.Tensor = batch["labels"].to(device)  # (batch_size, num_queries, 2)

        nme_iod = calc_nme(xy, labels, iod_indices, norm_type="iod")  # (batch_size, num_queries)
        nme_s = calc_nme(xy, labels, norm_type="size")  # (batch_size, num_queries)

        start_idx = i * batch_size

        for b in range(curr_batch):
            idx = start_idx + b
            if isinstance(dataset.dataset, WFLW):
                img_path = dataset.dataset._img_paths[i * batch_size + b].removeprefix(str(dataset.dataset.dir)).lstrip("/")
                result.img_paths.append(img_path)
            elif isinstance(dataset.dataset, FaceSynthetics):
                img_path = Path(dataset.dataset._get_base_path(idx)).stem
                result.img_paths.append(f"{img_path}{dataset.dataset.image_ext}")
            elif isinstance(dataset.dataset, IbugTest):
                img_path = dataset.dataset._images[idx].removeprefix(str(dataset.dataset.dir)).lstrip("/")
                result.img_paths.append(img_path)

        batch_slice = slice(start_idx, start_idx + curr_batch)
        result.labels[batch_slice] = labels.cpu()
        result.xy[batch_slice] = xy.cpu()
        try:
            result.cov[batch_slice] = preds.cov.params.cpu()
        except:
            pass
        result.nme_iod[batch_slice] = nme_iod.cpu()
        result.nme_s[batch_slice] = nme_s.cpu()

    return result

EVAL_NOTEBOOK_PATH = Path(__file__).parent / "notebooks" / "eval.ipynb"

def main():
    import argparse
    import papermill
    parser = argparse.ArgumentParser(description="Evaluate QLOT on image datasets")
    parser.add_argument("checkpoints", type=Path, nargs="+", help="Path(s) to the model checkpoint(s) to evaluate")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to the dataset directory")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Directory to save evaluation results")
    parser.add_argument("--copy-checkpoint", action="store_true", help="Copy the checkpoint to the output directory after evaluation")
    parser.add_argument("--notebook", type=Path, default=EVAL_NOTEBOOK_PATH, help="Path to the evaluation notebook (relative to `--notebook-cwd`)")
    parser.add_argument("--notebook-cwd", type=Path, default=Path("."), help="Current working directory for the notebook execution")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to append to the output filenames")
    args = parser.parse_args()

    checkpoint_path: Path
    dataset_dir: Path = args.dataset_dir.absolute()
    out_dir: Path = args.out_dir.absolute()
    cwd: Path = args.notebook_cwd.absolute()

    for i, checkpoint_path in enumerate(args.checkpoints):
        print(f"Evaluating checkpoint {i + 1}/{len(args.checkpoints)}: {checkpoint_path}")
        _, config = load(checkpoint_path, None, None, None)

        if args.copy_checkpoint:
            shutil.copy(checkpoint_path, args.out_dir / f"checkpoint_{config.run}{checkpoint_path.suffix}")

        # Run the evaluation notebook with papermill
        papermill.execute_notebook(
            input_path=args.notebook,
            output_path=args.out_dir / f"eval_{config.run}{args.suffix}.ipynb",
            parameters={
                "checkpoint_path": str(checkpoint_path.absolute()),
                "dataset_dir": str(dataset_dir),
                "out_dir": str(out_dir),
                "is_papermill": True,
                "out_suffix": args.suffix,
            },
            cwd=cwd
        )

if __name__ == "__main__":
    main()