import argparse
from time import time

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from tqdm import tqdm

from pfeife.mp import initialize_pfeife, get_state, PartialRunner, partial_compiler
from pfeife import PipeOption

from transformers import ViTConfig, ViTForImageClassification
from datasets import load_dataset


def format_num(num: int, bytes=False):
    """Scale bytes to its proper format, e.g. 1253656 => '1.20MB'"""
    factor = 1024 if bytes else 1000
    suffix = "B" if bytes else ""
    for unit in ["", " K", " M", " G", " T", " P"]:
        if num < factor:
            return f"{num:.2f}{unit}{suffix}"
        num /= factor


def run_vit(args, model, data_loader):
    # model = setup(args, model, data_loader)

    start_time = time()
    iter_cnt = 0
    warmup = 0

    last_time = time()
    total_iter = 0

    net_state = get_state()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate * net_state.world_size,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    option = PipeOption.from_args(args)
    runner = PartialRunner(option, optimizer)
    model = torch.compile(model, backend=partial_compiler)

    data_len = len(data_loader)
    batch = next(iter(data_loader))
    batch["labels"] = batch["labels"].to(net_state.device)

    with tqdm(range(data_len)) as pbar:
        for _ in pbar:

            def iter_fn(batch):
                outputs = model(pixel_values=batch["pixel_values"]).logits.view(-1, 10)
                labels = batch["labels"]
                loss = criterion(outputs, labels)
                loss.backward()
                return loss

            runner.set_exec_fn(iter_fn)
            losses = runner.step(batch)
            loss = sum([loss.item() for loss in losses])

            pbar.set_postfix({"loss": loss})

            if warmup < 20:
                warmup += 1
                start_time = time()
                last_time = time()
            else:
                iter_cnt += 1

            if warmup >= 20 and iter_cnt % 20 == 0:
                this_time = time()
                print(
                    f"throughput {iter_cnt}: {iter_cnt / (this_time - start_time):7.5f} it/s, {20 / (this_time - last_time):7.5f} it/s",
                    flush=True,
                )
                last_time = this_time

            total_iter += 1
            if total_iter >= args.iter_cnt:
                break

    end_time = time()

    print(f"throughput: {iter_cnt / (end_time - start_time):7.5f} it/s", flush=True)


def main():
    initialize_pfeife()

    parser = argparse.ArgumentParser()
    parser.add_argument("--iter_cnt", type=int, default=300)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight decay to use."
    )

    PipeOption.add_arguments(parser)

    args = parser.parse_args()
    print(args)

    batch_size = args.batch_size * args.batch_cnt

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def train_transforms(examples):
        examples["pixel_values"] = [
            transform(image.convert("RGB")) for image in examples["img"]
        ]
        return examples

    train_ds, test_ds = load_dataset("cifar10", split=["train", "test"])
    # split up training into training + validation
    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits["train"]

    train_ds.set_transform(train_transforms)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    # Prepare dataloader
    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    config = ViTConfig(
        hidden_size=1408,
        image_size=224,
        intermediate_size=int(1408 * (48 / 11)),
        num_attention_heads=16,
        qkv_bias=True,
        num_labels=10,
        num_channels=3,
        num_hidden_layers=32,
        patch_size=14,
        hidden_act="gelu",
        # problem_type="single_label_classification",
    )

    config.id2label = {
        id: label for id, label in enumerate(train_ds.features["label"].names)
    }
    config.label2id = {label: id for id, label in config.id2label.items()}

    model = ViTForImageClassification(config)

    run_vit(args, model, train_dataloader)


if __name__ == "__main__":
    main()
