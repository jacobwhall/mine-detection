# much of this code was adapted from
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from segmentation_models_pytorch.losses import (BINARY_MODE, DiceLoss,
                                                JaccardLoss)
from torch import cuda, nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ToTensor

from config import load_config
from dataset import MiningDataset

config = load_config()

tile_dir = Path(config["project_path"]) / config["tile_dir"]

if cuda.is_available():
    cuda.empty_cache()
    device = torch.device("cuda")
    print("Found a CUDA device!")
else:
    raise SystemError("Cuda device not available!")


def train_model(
    data_loaders,
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    since=time.time(),
):
    writer = SummaryWriter()
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        torch.save(model.state_dict(), best_model_params_path)

        best_acc = 0.0
        no_better_count = 0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 12)

            for phase in ["train", "validate"]:
                if phase == "train":
                    model.train()  # set model to training mode
                else:
                    model.eval()  # set model to evaluate mode

                running_loss: float = 0.0
                # running_corrects: int = 0
                running_tp: int = 0
                running_fp: int = 0
                running_fn: int = 0
                running_tn: int = 0

                for image, label in data_loaders[phase]:
                    # NOTE: these are for debugging / QA
                    h, w = image.shape[2:]
                    assert h % 32 == 0 and w % 32 == 0
                    assert label.max() <= 1.0 and label.min() >= 0

                    # send inputs and labels to GPU
                    image = image.to(device)
                    label = label.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        # outputs = a batch_size * 1 * tile_dim * tile_dim tensor of outputs
                        # these are all pretty random float values
                        outputs = model(image)

                        # torch.max function returns (max across dimension, indices)
                        # preds = a batch_size * 1 * tile_dim * tile_dim tensor of predicted class values (0 or 1)
                        # the single dimension (index 1) is only kept with keepdim
                        preds = (outputs.sigmoid() > 0.5).float()

                        # this is zero-dimension tensor, i.e. one float
                        loss = criterion(outputs, label)

                        # backward + optimize, if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # calculate statistics
                    # loss value (as float) * batch size
                    running_loss += loss.item() * image.size(0)

                    # (preds == label) depends:
                    # if keepdim above, it is of shape
                    # if not keepdim, it is much larger, 6 * 6 * 224 * 224
                    assert preds.shape == label.shape

                    tp, fp, fn, tn = smp.metrics.get_stats(
                        preds.long(), label.long(), mode=BINARY_MODE
                    )
                    running_tp += tp.sum()
                    running_fp += fp.sum()
                    running_fn += fn.sum()
                    running_tn += tn.sum()

                if phase == "train":
                    scheduler.step()

                # running_loss / number of image/label pairs in this phase's data loader
                epoch_loss = running_loss / dataset_sizes[phase]

                # print(f"running corrects: {running_corrects}")

                running_t = running_tp + running_tn
                running_f = running_fp + running_fn

                total_pixels = running_t + running_f
                assert (
                    total_pixels
                    == dataset_sizes[phase]
                    * config["tile_dimension"]
                    * config["tile_dimension"]
                )

                # running_corrects / number of pixels in this batch
                epoch_acc = running_t.double() / total_pixels
                if phase == "validate":
                    print("")

                print(f"{phase} loss: {epoch_loss:.4f}")
                print(f"{phase} acc: {epoch_acc * 100}%")
                print(f"True positives: {running_tp / total_pixels * 100}% of total")
                print(f"False positives: {running_fp / total_pixels * 100}% of total")
                print(f"True negatives: {running_tn / total_pixels * 100}% of total")
                print(f"False negatives: {running_fn / total_pixels * 100}% of total")

                running_metrics = (running_tp, running_fp, running_fn, running_tn)

                precision = smp.metrics.precision(*running_metrics)
                recall = smp.metrics.recall(*running_metrics)
                f1 = smp.metrics.f1_score(*running_metrics)
                b = 2
                f_beta = smp.metrics.fbeta_score(*running_metrics, beta=b)
                iou = smp.metrics.iou_score(*running_metrics)

                writer.add_scalar(f"loss {phase}", epoch_loss, epoch)
                writer.add_scalar(f"accuracy {phase}", epoch_acc, epoch)
                writer.add_scalar(f"tp {phase}", running_tp / total_pixels, epoch)
                writer.add_scalar(f"fp {phase}", running_fp / total_pixels, epoch)
                writer.add_scalar(f"tn {phase}", running_tn / total_pixels, epoch)
                writer.add_scalar(f"fn {phase}", running_fn / total_pixels, epoch)
                writer.add_scalar(f"precision {phase}", precision, epoch)
                writer.add_scalar(f"recall {phase}", recall, epoch)
                writer.add_scalar(f"f1 {phase}", f1, epoch)
                writer.add_scalar(f"weighted f1 (b={b}) {phase}", f_beta, epoch)
                writer.add_scalar(f"iou {phase}", iou, epoch)

                # deep copy the model if it's better than our previous best
                if phase == "validate" and epoch_acc > best_acc:
                    no_better_count = 0
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
                else:
                    no_better_count += 1

            if (
                config["no_better_timeout"] > 0
                and no_better_count >= config["no_better_timeout"]
            ):
                print(
                    "Stopping early, no better validation score for {} epochs".format(
                        config["no_better_timeout"]
                    )
                )
                break

            print()

        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val Acc: {best_acc:4f}")
        # f1 accuracy
        # confusion matrices
        # IoU (intersection over union)

        model.load_state_dict(torch.load(best_model_params_path))

    writer.flush()
    writer.close()

    return model


if __name__ == "__main__":
    model = smp.Unet(
        encoder_name=config["backbone"],
        encoder_weights=config["encoder_weights"],
        in_channels=3,
        classes=1,
    ).to(device)

    # criterion = JaccardLoss(BINARY_MODE)
    criterion = DiceLoss(BINARY_MODE)

    data_loader_kwargs = {
        "batch_size": config["batch_size"],
        "shuffle": True,  # this is redundant, tiles are already shuffled
        "num_workers": 6,
    }

    transform = Compose(
        [
            get_preprocessing_fn(
                config["backbone"], pretrained=config["encoder_weights"]
            ),
            lambda a: a.astype("float32"),
            ToTensor(),
        ]
    )

    datasets = {
        "train": MiningDataset(tile_dir=(tile_dir / "train"), transform=transform),
        "validate": MiningDataset(
            tile_dir=(tile_dir / "validate"), transform=ToTensor()
        ),
    }
    data_loaders = {
        x: DataLoader(
            datasets[x],
            **data_loader_kwargs,
        )
        for x in ["train", "validate"]
    }

    dataset_sizes = {
        x: len(datasets[x]) for x in ["train", "validate"]
    }  # dict containing count of image/label pairs for each dataset

    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.92)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_model(
        data_loaders, model, criterion, optimizer, scheduler, config["num_epochs"]
    )
