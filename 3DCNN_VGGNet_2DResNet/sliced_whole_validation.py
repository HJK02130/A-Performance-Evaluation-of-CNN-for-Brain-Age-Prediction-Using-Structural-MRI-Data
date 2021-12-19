import json

from dataset import PAC2019, PAC20192D
from model import Model, VGGBasedModel, VGGBasedModel2D
from model_resnet import resnet18

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import medicaltorch.transforms as mt_transforms
import torchvision as tv
import torchvision.utils as vutils

import matplotlib.pyplot as plt
from collections import defaultdict, Counter

from tqdm import *


with open("config.json") as fid:
    ctx = json.load(fid)

val_set = PAC2019(ctx, set='val', split=0.8)

val_loader = DataLoader(val_set, shuffle=False, drop_last=False,
                             num_workers=8, batch_size=1)

model = resnet18()
model.cuda()
# model.load_state_dict(torch.load('models/lr0.0006_rampup20.pt'))
model.load_state_dict(torch.load('models/2d.pt'))
model.eval()

portion = 0.8
errors = []
error_per_age = defaultdict(list)
error_per_age_per_slice = defaultdict(lambda: defaultdict(list))
errors_val = []
for i, data in enumerate(tqdm(val_loader)):
    gm_image = Variable(data["gm"]).float().cuda()
    wm_image = Variable(data["wm"]).float().cuda()
    # print(input_image.shape)


    slices = []
    start = int((1.-portion)*gm_image.shape[1])
    end = int(portion*gm_image.shape[1])
    gm_image = gm_image[0,start:end,:,:]
    wm_image = wm_image[0,start:end,:,:]
    # print(gm_image.shape)
    for slice_idx in range(gm_image.shape[0]):
        slice_gm = gm_image[slice_idx,:,:]
        slice_gm = slice_gm.unsqueeze(0)
        slice_wm = wm_image[slice_idx,:,:]
        slice_wm = slice_wm.unsqueeze(0)
        slice = torch.cat([slice_gm, slice_wm], dim=0)
        # print(slice.shape)
        slices.append({
            'image': slice,
            'label': data['label']
        })
        # print('Slice: ', slice.shape)

    error = []
    for idx, slice in enumerate(slices):
        age = int(slice['label'].item())
        slice['image'] = slice['image'].unsqueeze(0)
        # print(slice['image'].shape)
        output = model(slice['image'])
        # print(output[0], slice['label'])
        error.append(np.abs(output[0].item() - slice['label'].item()))
        error_per_age_per_slice[idx][age].append(np.abs(output[0].item() - slice['label'].item()))
    # print(error)
    errors.append(error)
    errors_val.append(np.mean(error))
    error_per_age[int(slice['label'].item())].append(np.mean(error))

print('Validation error: ', np.mean(errors_val))
min_slice = 0
# print(error_per_age_per_slice.keys())
max_slice = len(error_per_age_per_slice.keys())
min_age = min(error_per_age_per_slice[0].keys())
max_age = max(error_per_age_per_slice[0].keys())+1
# print('Min/max: ', min_age, max_age)
heatmap = np.zeros((max_age, max_slice))
# print(error_per_age_per_slice.keys())
# print(error_per_age_per_slice[0].keys())
# print(list(sorted(error_per_age_per_slice[0].keys())))
for slice_idx in sorted(error_per_age_per_slice.keys()):
    # print('here')
    for age in range(0, 75):
        # print('age: here')
        # print('Slice/Age: %d/%d --> ' % (slice_idx, age), error_per_age_per_slice[slice_idx][age])
        mean = np.mean(error_per_age_per_slice[slice_idx][age])
        if not np.isnan(mean):
            heatmap[age,slice_idx] = mean
        # print('mean: ', np.mean(error_per_age_per_slice[slice_idx][age]))
plt.imshow(heatmap, cmap='viridis')
plt.colorbar()
plt.ylabel('Age')
plt.xlabel('Slice')
# plt.grid()
plt.show()
# raise
# print(error_per_age)
sorted_values = []
keys = []
for k in sorted(error_per_age.keys()):
    sorted_values.append(error_per_age[k])
    keys.append(k)

fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)

ax.boxplot(sorted_values)
ax.set_xticklabels(keys)
plt.show()


errors = np.array(errors)
# print(errors.shape)
mean_errors = np.mean(errors, axis=0)
# plt.plot(mean_errors)
fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)
x = np.linspace(0, errors.shape[1])
extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2.,0,1]
ax.imshow(mean_errors[np.newaxis,:], cmap="viridis", aspect="auto", extent=extent)
# print(mean_errors.shape)
# print(x.shape)
ax2.plot(np.arange(mean_errors.shape[0]),mean_errors)

plt.ylabel('Mean Absolute Error (MAE)')
plt.xlabel('Slice index')

plt.show()
# print(mean_errors)
