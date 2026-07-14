from __future__ import annotations

import argparse
import json
import math
import os
import random
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from accelerate import Accelerator
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer


def pil_to_tensor_rgb(img: Image.Image, size: int) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return x


def pil_mask_to_tensor(mask: Image.Image, size: int) -> torch.Tensor:
    mask = mask.convert("L").resize((size, size), Image.NEAREST)
    arr = np.array(mask).astype(np.float32) / 255.0
    arr = (arr > 0.5).astype(np.float32)
    x = torch.from_numpy(arr).unsqueeze(0)
    return x


class WoodInpaintDataset(Dataset):
    def __init__(
        self,
        image_root: str,
        mask_root: str,
        metadata_root: Optional[str] = None,
        resolution: int = 512,
        empty_prob: float = 0.05,
        class_name: str = "crack",
        prompt: str = "natural crack on wood surface",
        max_samples: Optional[int] = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.mask_root = Path(mask_root)
        self.metadata_root = Path(metadata_root) if metadata_root is not None else None
        self.resolution = resolution
        self.empty_prob = empty_prob
        self.class_name = class_name
        self.prompt = prompt
        self.max_samples = max_samples

        class_img_dir = self.image_root / self.class_name
        class_mask_dir = self.mask_root / self.class_name
        class_meta_dir = self.metadata_root / self.class_name if self.metadata_root else None

        if not class_img_dir.exists():
            raise RuntimeError(f"Image directory not found: {class_img_dir}")
        if not class_mask_dir.exists():
            raise RuntimeError(f"Mask directory not found: {class_mask_dir}")
        if class_meta_dir is not None and not class_meta_dir.exists():
            print(f"[WARN] Metadata directory not found, continuing without metadata: {class_meta_dir}")
            class_meta_dir = None

        self.samples: list[dict] = []

        image_files = sorted(
            [
                p for p in class_img_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            ]
        )

        for img_path in image_files:
            mask_path = class_mask_dir / img_path.name
            if not mask_path.exists():
                continue

            meta_path = None
            if class_meta_dir is not None:
                candidate = class_meta_dir / f"{img_path.stem}.json"
                if candidate.exists():
                    meta_path = candidate

            self.samples.append(
                {
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "meta_path": meta_path,
                    "prompt": self.prompt,
                    "class_name": self.class_name,
                }
            )

        if not self.samples:
            raise RuntimeError(
                f"No matching image/mask pairs found for class '{self.class_name}'."
            )

        if self.max_samples is not None and self.max_samples > 0:
            total = len(self.samples)
            self.samples = self.samples[: self.max_samples]
            print(
                f"Limiting dataset from {total} to {len(self.samples)} samples "
                f"(--max_samples={self.max_samples})."
            )

        print(f"Loaded {len(self.samples)} training pairs for class '{self.class_name}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        image = Image.open(sample["image_path"])
        mask = Image.open(sample["mask_path"])

        image_t = pil_to_tensor_rgb(image, self.resolution)
        mask_t = pil_mask_to_tensor(mask, self.resolution)

        if random.random() < 0.5:
            noise = torch.randn_like(mask_t) * 0.05
            mask_t = (mask_t + noise).clamp(0, 1)

        if random.random() < 0.5:
            mask_t = F.max_pool2d(mask_t, kernel_size=3, stride=1, padding=1)

        if random.random() < self.empty_prob:
            mask_t = torch.zeros_like(mask_t)

        masked_image_t = image_t * (1.0 - mask_t)

        return {
            "image": image_t,
            "mask": mask_t,
            "masked_image": masked_image_t,
            "prompt": sample["prompt"],
            "class_name": sample["class_name"],
            "image_name": sample["image_path"].name,
        }


def collate_fn(examples: list[dict], tokenizer: CLIPTokenizer) -> dict:
    pixel_values = torch.stack([e["image"] for e in examples])
    masks = torch.stack([e["mask"] for e in examples])
    masked_images = torch.stack([e["masked_image"] for e in examples])
    prompts = [e["prompt"] for e in examples]
    image_names = [e["image_name"] for e in examples]

    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )

    return {
        "pixel_values": pixel_values,
        "masks": masks,
        "masked_images": masked_images,
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
        "prompts": prompts,
        "image_names": image_names,
    }


def build_validation_canvas(size: int = 512) -> tuple[Image.Image, Image.Image]:
    val_img = Image.new("RGB", (size, size), (184, 153, 112))
    val_mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(val_mask)

    # resin-like irregular blob
    points = [(210, 210), (245, 185), (300, 200), (320, 245), (300, 290), (250, 305), (205, 275), (190, 235)]
    draw.polygon(points, fill=255)

    return val_img, val_mask


def save_validation(
    accelerator: Accelerator,
    unet,
    vae,
    text_encoder,
    tokenizer,
    base_model: str,
    out_dir: str,
    step: int,
    prompt: str,
    suffix: str,
) -> None:
    if not accelerator.is_main_process:
        return

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        base_model,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        safety_checker=None,
        torch_dtype=torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32,
    ).to(accelerator.device)

    pipe.set_progress_bar_config(disable=True)

    val_img, val_mask = build_validation_canvas(512)
    generator = torch.Generator(device=accelerator.device).manual_seed(1234)

    with torch.no_grad():
        out = pipe(
            prompt=prompt,
            image=val_img,
            mask_image=val_mask,
            guidance_scale=7.0,
            num_inference_steps=30,
            generator=generator,
        ).images[0]

    validation_dir = Path(out_dir) / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    out.save(validation_dir / f"step_{step:07d}_{suffix}.png")

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_training_config(args: argparse.Namespace, sample_count: int) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["sample_count"] = sample_count
    (out / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-inpainting",
    )
    parser.add_argument(
        "--train_root",
        type=str,
        default="./wood_inpaint_from_det_train",
        help="Root folder containing images/ masks/ and optionally metadata/",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="crack",
        help="Target class to train inpainting on",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="natural crack on wood surface",
        help="Prompt used during training/validation",
    )
    parser.add_argument("--output_dir", type=str, default="./crack_inpaint_lora_out")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--empty_prob", type=float, default=0.05)

    parser.add_argument(
        "--max_samples",
        type=int,
        default=175,
        help="Limit the dataset to the first N pairs (after sorting). Use 0 or a negative value for no limit.",
    )

    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None,
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    image_root = Path(args.train_root) / "images"
    mask_root = Path(args.train_root) / "masks"
    metadata_root = Path(args.train_root) / "metadata"

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()

    if accelerator.is_main_process:
        unet.print_trainable_parameters()

    max_samples_arg = args.max_samples if args.max_samples and args.max_samples > 0 else None

    dataset = WoodInpaintDataset(
        image_root=str(image_root),
        mask_root=str(mask_root),
        metadata_root=str(metadata_root) if metadata_root.exists() else None,
        resolution=args.resolution,
        empty_prob=args.empty_prob,
        class_name=args.class_name,
        prompt=args.class_prompt,
        max_samples=max_samples_arg,
    )

    if accelerator.is_main_process:
        save_training_config(args, len(dataset))

    train_dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    max_train_steps = 2000
    args.num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=torch.float32)

    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")

    for epoch in range(args.num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float32)
                masks = batch["masks"].to(accelerator.device, dtype=torch.float32)
                masked_images = batch["masked_images"].to(accelerator.device, dtype=torch.float32)
                input_ids = batch["input_ids"].to(accelerator.device)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    masked_latents = vae.encode(masked_images).latent_dist.sample()
                    masked_latents = masked_latents * vae.config.scaling_factor

                    encoder_hidden_states = text_encoder(input_ids)[0]

                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                    dtype=torch.long,
                )

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                latent_mask = F.interpolate(
                    masks,
                    size=(latents.shape[-2], latents.shape[-1]),
                    mode="nearest",
                )

                model_input = torch.cat([noisy_latents, latent_mask, masked_latents], dim=1)

                model_pred = unet(
                    model_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unsupported prediction type: {noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=float(loss.detach().item()),
                    lr=lr_scheduler.get_last_lr()[0],
                )

                if global_step % args.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_dir, exist_ok=True)
                        accelerator.unwrap_model(unet).save_pretrained(save_dir)

                if global_step % args.validation_steps == 0:
                    accelerator.wait_for_everyone()
                    save_validation(
                        accelerator=accelerator,
                        unet=unet,
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        base_model=args.pretrained_model_name_or_path,
                        out_dir=args.output_dir,
                        step=global_step,
                        prompt=args.class_prompt,
                        suffix=args.class_name,
                    )

            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final_lora")
        os.makedirs(final_dir, exist_ok=True)
        accelerator.unwrap_model(unet).save_pretrained(final_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()