import sys
import glob
import os.path
import numpy as np
import pandas as pd
from subimage_inspector import Subimage_inspector

from ast import literal_eval
import argparse
from torch.utils.data import DataLoader
from image_loader import Data_Set, Batch_Sampler

def cell_embeddings(model, model_path, images_folder, centers, inspection_file, channel_names, channel_substrings, num_workers):
    output = []
    if model == 'dino4cells':
        from dino4cells.simple_embed import dino_model
        model = dino_model(args.model_path)
        input_channels = ['DNA', 'RNA', 'ER', 'AGP', 'Mito']
    else:
        from cpcnn.simple_embed import cpcnn_model
        model = cpcnn_model(args.model_path)
        input_channels = ['DNA', 'RNA', 'ER', 'AGP', 'Mito']

    # generate image names
    missing_channels = set(input_channels) - set(channel_names)
    if len(missing_channels) > 0:
        raise Exception('ERROR: Some channel names are missing. This model requires ' + ','.join(input_channels))
    if 'DNA' not in input_channels:
        raise Exception('ERROR: The DNA channel should be specified')
    input_channels

    dna_images = [images_folder + imname for imname in centers.index]
    channel_filters = dict(zip(channel_names, channel_substrings))
    other_images = [[dnaim.replace(channel_filters['DNA'], channel_filters[channel]) for dnaim in dna_images]
                     for channel in input_channels if channel != 'DNA']
    image_groups = list(zip(dna_images, *other_images))

    # check that image paths point to actual files
    actual_files = set(glob.glob(images_folder + '*'))
    image_groups = [imgrp for imgrp in image_groups if all(filepath in actual_files for filepath in imgrp)]
    if len(image_groups) != len(centers):
        print(f"WARNING: Found complete image sets for {len(image_groups)} out of {len(centers)}")

    ds = Data_Set(image_groups, centers)
    bs = Batch_Sampler(image_groups, centers)
    dataloader = DataLoader(ds, batch_sampler=bs, num_workers=num_workers, pin_memory=True)

    if inspection_file is not None:
        num_sample_crops = 100
        num_channels = len(input_channels)
        num_cells = len(ds)
        resolution = 128
        subimage_inspector = Subimage_inspector(num_cells, num_sample_crops, num_channels, resolution)

    # generate embeddings
    for dna_imnames, centers_i, centers_j, subimages in dataloader:
        dna_imname = dna_imnames[0]
        centers_i = centers_i.tolist()
        centers_j = centers_j.tolist()
        print(dna_imname, len(centers_i))
        if inspection_file is not None:
            subimage_inspector.add(dna_imname, centers_i, centers_j, subimages)
        embeddings = model(subimages)
        for i, j, em in zip(centers_i, centers_j, embeddings):
            output.append([dna_imname, i, j, em])
    
    if inspection_file is not None:
        subimage_inspector.save(inspection_file)
    return output


parser = argparse.ArgumentParser(description='run_dino4cells', prefix_chars='@')
parser.add_argument('model', type=str, choices=['dino4cells', 'cpcnn'])
parser.add_argument('model_path', type=str, help='model')
parser.add_argument('plate_path', type=str, help='folder containing images')
parser.add_argument('channel_names', type=str, help='comma seperated names of channels')
parser.add_argument('channel_substrings', type=str, help='comma seperated substrings of filename to identify channels')
parser.add_argument('centers_path', type=str, help='filename with cell centers')
parser.add_argument('num_workers', type=int, help='number of processes for loading data')
parser.add_argument('output_file', type=str, help='output filename', nargs='?', default='embedding.tsv')
parser.add_argument('inspection_file', type=str, help='output filename with image crops for manual inspection', nargs='?')
args = parser.parse_args()

images_folder = args.plate_path
if not images_folder.endswith('/'):
    images_folder = images_folder + '/'


if args.channel_names.count(',') != args.channel_substrings.count(','):
    raise Exception('ERROR: Channel names and substrings should have the same length.')

channel_names      = [s.strip() for s in args.channel_names.split(',')]
channel_substrings = [s.strip() for s in args.channel_substrings.split(',')]

centers = pd.read_table(args.centers_path, converters={'i':literal_eval, 'j':literal_eval}, index_col='file')

output = cell_embeddings(args.model, args.model_path, images_folder, centers, args.inspection_file, channel_names, channel_substrings, args.num_workers)

df = pd.DataFrame(output, columns="file i j embedding".split())
df.to_csv(args.output_file, sep="\t", index=None)