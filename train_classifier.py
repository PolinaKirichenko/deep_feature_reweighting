import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import tqdm
import argparse
import sys
from collections import defaultdict
import json
from functools import partial

from wb_data import WaterBirdsDataset, get_loader, get_transform_cub, log_data

from utils import MultiTaskHead
from utils import Logger, AverageMeter, set_seed, evaluate, get_y_p
from utils import update_dict, get_results, write_dict_to_tb


parser = argparse.ArgumentParser(description="Train model on waterbirds data")
parser.add_argument(
    "--data_dir", type=str,
    default=None,
    help="Train dataset directory")
parser.add_argument(
    "--test_wb_dir", type=str,
    default=None,
    help="Test data directory, regular waterbirds")
parser.add_argument(
    "--test_grey_dir", type=str,
    default=None,
    help="Test data directory, waterbirds w/o background")
parser.add_argument(
    "--test_places_dir", type=str,
    default=None,
    help="Test data directory, places")
parser.add_argument(
    "--output_dir", type=str,
    default="logs/",
    help="Output directory")

parser.add_argument("--pretrained_model", action='store_true', help="Use pretrained model")
parser.add_argument("--reweight_classes", action='store_true', help="Reweight classes")
parser.add_argument("--reweight_places", action='store_true', help="Reweight based on place")
parser.add_argument("--reweight_groups", action='store_true', help="Reweight groups")
parser.add_argument("--augment_data", action='store_true', help="Train data augmentation")
parser.add_argument("--scheduler", action='store_true', help="Learning rate scheduler")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum_decay", type=float, default=0.9)
parser.add_argument("--init_lr", type=float, default=0.001)
parser.add_argument("--eval_freq", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)

# Target
parser.add_argument("--multitask", action='store_true', help="Predict label and group")
parser.add_argument("--predict_place", action='store_true', help="Predict label and group")

#Understanding exps
# parser.add_argument("--no_minority_groups", action='store_true',
#                     help="Remove all minority group examples from the train data")
parser.add_argument("--num_minority_groups_remove", type=int, required=False, default=0)

parser.add_argument("--resume", type=str, default=None)


args = parser.parse_args()

assert args.reweight_groups + args.reweight_classes <= 1
assert args.multitask + args.predict_place <= 1

print('Preparing directory %s' % args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
    args_json = json.dumps(vars(args))
    f.write(args_json)

set_seed(args.seed)

writer = SummaryWriter(log_dir=args.output_dir)
logger = Logger(os.path.join(args.output_dir, 'log.txt'))

splits = ["train", "test", "val"]
basedir = args.data_dir

# Data
target_resolution = (224, 224)
train_transform = get_transform_cub(target_resolution=target_resolution, train=True, augment_data=args.augment_data)
test_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=args.augment_data)

trainset = WaterBirdsDataset(basedir=basedir, split="train", transform=train_transform)
testset_dict = {
    'wb': WaterBirdsDataset(basedir=args.test_wb_dir, split="test", transform=test_transform),
    'wb_val': WaterBirdsDataset(basedir=args.test_wb_dir, split="val", transform=test_transform),
}

if not args.predict_place and not (args.test_grey_dir is None):
    testset_dict['grey'] = WaterBirdsDataset(basedir=args.test_grey_dir, split="test", transform=test_transform)
if ((args.predict_place) and not (args.test_places_dir is None)) or args.multitask:
    testset_dict['places'] = WaterBirdsDataset(basedir=args.test_places_dir, split="test", transform=test_transform)

if args.num_minority_groups_remove > 0:
    print("Removing minority groups")
    print("Initial groups", np.bincount(trainset.group_array))
    group_counts = trainset.group_counts
    minority_groups = np.argsort(group_counts.numpy())[:args.num_minority_groups_remove]
    minority_groups
    idx = np.where(np.logical_and.reduce(
        [trainset.group_array != g for g in minority_groups], initial=True))[0]
    trainset.y_array = trainset.y_array[idx]
    trainset.group_array = trainset.group_array[idx]
    trainset.confounder_array = trainset.confounder_array[idx]
    trainset.filename_array = trainset.filename_array[idx]
    trainset.metadata_df = trainset.metadata_df.iloc[idx]
    print("Final groups", np.bincount(trainset.group_array))

# testset = WaterBirdsDataset(basedir=basedir, split="test", transform=test_transform)
# valset = WaterBirdsDataset(basedir=basedir, split="val", transform=test_transform)

loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
train_loader = get_loader(
    trainset, train=True, reweight_groups=args.reweight_groups,
    reweight_classes=args.reweight_classes, reweight_places=args.reweight_places, **loader_kwargs)
test_loader_dict = {}
for test_name, testset_v in testset_dict.items():
    test_loader_dict[test_name] = get_loader(
        testset_v, train=False, reweight_groups=None,
        reweight_classes=None, reweight_places=None, **loader_kwargs)

# test_loader = get_loader(
#   testset, train=False, reweight_groups=None, reweight_classes=None, **loader_kwargs)

get_yp_func = partial(get_y_p, n_places=trainset.n_places)
log_data(logger, trainset, testset_dict['wb'], get_yp_func=get_yp_func)

# Model
n_classes = trainset.n_classes
model = torchvision.models.resnet50(pretrained=args.pretrained_model)
d = model.fc.in_features
if not args.multitask:
    model.fc = torch.nn.Linear(d, n_classes)
else:
    model.fc = MultiTaskHead(d, [n_classes, trainset.n_places])

# TODO: fix resuming from a checkpoint
if args.resume is not None:
    print('Resuming from checkpoint at {}...'.format(args.resume))
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)

model.cuda()

optimizer = torch.optim.SGD(
    model.parameters(), lr=args.init_lr, momentum=args.momentum_decay, weight_decay=args.weight_decay)
if args.scheduler:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs)
else:
    scheduler = None

criterion = torch.nn.CrossEntropyLoss()

logger.flush()

# Train loop
for epoch in range(args.num_epochs):
    model.train()
    loss_meter = AverageMeter()
    acc_groups = {g_idx : AverageMeter() for g_idx in range(trainset.n_groups)}
    if args.multitask:
        acc_place_groups = {g_idx: AverageMeter() for g_idx in range(trainset.n_groups)}

    for batch in tqdm.tqdm(train_loader):
        x, y, g, p = batch
        x, y, p = x.cuda(), y.cuda(), p.cuda()
        if args.predict_place:
            y = p

        optimizer.zero_grad()
        logits = model(x)
        if args.multitask:
            logits, logits_place = logits
            loss = criterion(logits, y) + criterion(logits_place, p)
            update_dict(acc_place_groups, p, g, logits_place)
        else:
            loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss, x.size(0))
        update_dict(acc_groups, y, g, logits)

    if args.scheduler:
        scheduler.step()
    logger.write(f"Epoch {epoch}\t Loss: {loss_meter.avg}\n")
    results = get_results(acc_groups, get_yp_func)
    logger.write(f"Train results \n")
    logger.write(str(results) + "\n")
    tag = "places_" if args.predict_place else ""
    write_dict_to_tb(writer, results, "train/" + tag, epoch)

    if args.multitask:
        results_place = get_results(acc_place_groups, get_yp_func)
        logger.write(f"Train place prediction results \n")
        logger.write(str(results_place) + "\n")
        write_dict_to_tb(writer, results_place, "train/places_", epoch)

    images = x[:4]
    # np.save(os.path.join(args.output_dir, "data.npy"), images.detach().cpu().numpy())
    # TODO: fix data visualization in tensorboard
    images_concat = torchvision.utils.make_grid(
        images, nrow=2, padding=2, pad_value=1.)
    writer.add_image("data/", images_concat, epoch)

    if epoch % args.eval_freq == 0:
        # Iterating over datasets we test on
        for test_name, test_loader in test_loader_dict.items():
            results = evaluate(model, test_loader, get_yp_func, args.multitask, args.predict_place)
            if args.multitask and test_name == "wb":
                results, results_places = results
                write_dict_to_tb(writer, results_places, "test_wb_places/", epoch)
                logger.write("Test results \n")
                logger.write(str(results))
            elif args.multitask:
                results, _ = results
            tag = test_name
            if test_name == "wb":
                tag = "wb_birds" if not args.predict_place else "wb_places"
            write_dict_to_tb(writer, results, "test_{}/".format(tag), epoch)
            logger.write("Test results \n")
            logger.write(str(results))

        torch.save(
            model.state_dict(), os.path.join(args.output_dir, 'tmp_checkpoint.pt'))

    logger.write('\n')

torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_checkpoint.pt'))
