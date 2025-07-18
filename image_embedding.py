import os
import sys
import glob
import argparse
import pandas as pd

from torch.utils.data import DataLoader
from utils.image_loader import Image_Data_Set


def image_embeddings(model_name, model_path, images_folder, output_file, channel_names, channel_substrings, num_workers, num_processes, process_idx):
    output = []
    if model_name == 'cellpose':
        from models.cellpose import cell_center_model
        model = cell_center_model()
        input_channels = ['DNA']
        num_output_features = 2
        csv_header = 'file i j'.split()
        collate_fn = lambda x : x[0] # avoids conversion to Torch tensor
        target_size = (512, 512)
        log_scale = True
        output_hdf5_supported = False # number of rows is not known beforehand
    elif model_name == 'unidino':
        from models.unidino import unidino_model
        model = unidino_model(args.model_path)
        input_channels = sorted(channel_names)
        num_output_features = 384 * len(input_channels)
        csv_header = ['file'] + [f'feature_{idx}' for idx in range(num_output_features)]
        collate_fn = None
        target_size = None
        log_scale = False
        output_hdf5_supported = True

    missing_channels = set(input_channels) - set(channel_names)
    if len(missing_channels) > 0:
        raise Exception('ERROR: Some channel names are missing. The following channels are required: ' + ','.join(input_channels))
    if len(channel_names) == 0:
        raise Exception('ERROR: please pass a list of channel names.')

    # create file groups
    channel_filters = dict(zip(channel_names, channel_substrings))
    files = set(glob.glob(f'{images_folder}*'))

    file_groups = [[file.replace(channel_substrings[0], channel_filters[channel]) for channel in input_channels]
                    for file in files if channel_substrings[0] in file]

    # check that the input files are complete
    num_groups = len(file_groups)
    file_groups = [group for group in file_groups if all(f in files for f in group)]
    if num_groups < len(file_groups):
        print(f"WARNING: Found complete image sets for {num_groups} out of {len(file_groups)}", file=sys.stderr)

    file_groups.sort(key = lambda x : x[0])
    file_groups = file_groups[process_idx::num_processes]

    ds = Image_Data_Set(file_groups, target_size, log_scale)
    dataloader = DataLoader(ds, batch_size=1, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    # set up hdf5, tsv, csv, and png output
    if num_processes > 1:
        # add process index to filename (e.g., turn output.tsv into output_3_4.tsv for process 3 out of 4)
        output_file = output_file.split('.')
        max_len = len(str(num_processes-1))
        output_file[-2] += f'_{process_idx:0{max_len}d}_{num_processes}'
        output_file = '.'.join(output_file)
    if output_file.endswith('.h5') or output_file.endswith('.hdf5'):
        assert(output_hdf5_supported)
        from utils.hdf5writer import embedding_writer
        num_rows = len(ds)
        writer = embedding_writer(output_file, model_name, num_rows, num_output_features, 'f4')
    else:
        from utils.csvwriter import CSVWriter
        writer = CSVWriter(output_file)
        if num_processes == 1 or process_idx == 0:
            writer.write_header(csv_header)

    # generate embeddings
    for filenames, im_size, images in dataloader:
        if collate_fn is None:
            filenames = filenames[0]
            im_size = im_size[0]
            images = images[0, ::]

        filename = filenames[0].removeprefix(images_folder) # name of the first channel
        print(filename, end='')
        embedding = model(images, im_size)
        if model_name == 'cellpose':
            print('', len(embedding[0]))
        else:
            # other models have only one embedding per image, not worth showing
            print()
        writer.writerow(filename, embedding)
    writer.close()

parser = argparse.ArgumentParser(description='per cell embedding', prefix_chars='@')
parser.add_argument('model', type=str, choices=['cellpose', 'unidino'])
parser.add_argument('model_path', type=str, help='model')
parser.add_argument('plate_path', type=str, help='folder containing images')
parser.add_argument('channel_names', type=str, help='comma seperated names of channels')
parser.add_argument('channel_substrings', type=str, help='comma seperated substrings of filename to identify channels')
parser.add_argument('num_workers', type=int, help='number of processes for loading data')
parser.add_argument('num_processes', type=int, help='number of parallel runs of this script', nargs='?', default=1)
parser.add_argument('process_idx', type=int, help='index of parallel run', nargs='?', default=0)
parser.add_argument('output_file', type=str, help='output filename', nargs='?', default='embedding.tsv')
args = parser.parse_args()

images_folder = args.plate_path
if not images_folder.endswith('/'):
    images_folder = images_folder + '/'


assert args.num_processes > 0 
assert args.process_idx < args.num_processes

if args.channel_names.count(',') != args.channel_substrings.count(','):
    raise Exception('ERROR: Channel names and substrings should have the same length.')

channel_names      = [s.strip() for s in args.channel_names.split(',')]
channel_substrings = [s.strip() for s in args.channel_substrings.split(',')]

image_embeddings(args.model, args.model_path, images_folder, args.output_file, channel_names, channel_substrings, args.num_workers, args.num_processes, args.process_idx)