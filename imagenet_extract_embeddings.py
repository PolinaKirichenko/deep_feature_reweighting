"""Extract embeddings from an ImageNet-like dataset split and save to disk.
"""

import torch
import torchvision
import numpy as np
import tqdm
import argparse
import os
import imagenet_datasets

parser = argparse.ArgumentParser(
    description="Extract embeddings from an ImageNet-like dataset split and "
              "save to disk.")
parser.add_argument(
    "--dataset", type=str, default="imagenet",
    help="Dataset variation [imagenet | imagenet-a | imagenet-r | imagenet-c "
         "| bg_challenge]")
parser.add_argument(
    "--dataset_dir", type=str, default="/datasets/imagenet-stylized/",
    help="ImageNet dataset directory")
parser.add_argument(
    "--split", type=str, default="train",
    help="ImageNet dataset directory")
parser.add_argument(
    "--model", type=str, default="resnet50",
    help="Model to use [resnet50 | vitb16]")
parser.add_argument(
    "--batch_size", type=int, default=500,
    help="Batch size")


args = parser.parse_args()


if args.model == "resnet50":
    model = torchvision.models.resnet50(pretrained=True).cuda()
    def get_embed(m, x):
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)

        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)

        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    resize_size, crop_size = 256, 224


elif args.model == "vitb16":
    from pytorch_pretrained_vit import ViT
    model = ViT('B_16_imagenet1k', pretrained=True).cuda()

    def get_embed(m, x):
        b, c, fh, fw = x.shape
        x = m.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        if hasattr(m, 'class_token'):
            x = torch.cat((m.class_token.expand(b, -1, -1), x), dim=1)
        if hasattr(m, 'positional_embedding'):
            x = m.positional_embedding(x)
        x = m.transformer(x)
        if hasattr(m, 'pre_logits'):
            x = m.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(m, 'fc'):
            x = m.norm(x)[:, 0]
        return x
    resize_size, crop_size = 384, 384
else:
    raise ValueError("Unknown model")

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(resize_size),
    torchvision.transforms.CenterCrop(crop_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
])


if torch.cuda.device_count() > 1:
    raise NotImplementedError
model.eval()

ds, loader = imagenet_datasets.get_imagenet_like(
    name=args.dataset,
    datapath=args.dataset_dir,
    split=args.split,
    transform=transform,
    batch_size=args.batch_size,
    shuffle=False
)

all_embeddings = []
all_y = []

not_printed = True
for x, y in tqdm.tqdm(loader):
    with torch.no_grad():
        embed = get_embed(model, x.cuda()).detach().cpu().numpy() 
        all_embeddings.append(embed)
        all_y.append(y.detach().cpu().numpy())
        if not_printed:
            print("Embedding shape:", embed.shape)
            not_printed = False

all_embeddings = np.vstack(all_embeddings)
all_y = np.concatenate(all_y)

np.savez(os.path.join(
        args.dataset_dir,
        f"{args.dataset}_{args.model}_{args.split}_embeddings.npz"),
    embeddings=all_embeddings,
    labels=all_y)
