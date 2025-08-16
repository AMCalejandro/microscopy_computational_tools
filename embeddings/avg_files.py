import h5py
import numpy as np
from glob import glob
import glob
import os
import argparse



def compute_well_averages(input_path, output_path):
    with h5py.File(input_path, 'r') as h5_file:
        features = h5_file['deepprofiler'][()]
        meta_group = h5_file['meta']
        metadata = {}
        for key in meta_group.keys():
            metadata[key] = meta_group[key][()]
            if isinstance(metadata[key][0], bytes):
                metadata[key] = [v.decode('utf-8') for v in metadata[key]]

    well_indices = features[:, 0].astype(int)
    feature_data = features[:, 3:]
    unique_wells = np.unique(well_indices)
    averaged_features = np.zeros((len(unique_wells), feature_data.shape[1]))
    for i, well_idx in enumerate(unique_wells):
        mask = well_indices == well_idx
        averaged_features[i] = np.mean(feature_data[mask], axis=0)
    
    well_names = [metadata['Metadata_Well'][idx] for idx in unique_wells]
    with h5py.File(output_path, 'w') as h5_out:
        h5_out.create_dataset('deepprofiler_mean', data=averaged_features)
        meta_group = h5_out.create_group('meta')
        well_names_bytes = [name.encode('utf-8') for name in well_names]
        meta_group.create_dataset('Metadata_Well', data=well_names_bytes)


def merge_avg_h5_stream(tmp_paths, output_path):
    """
    Merge multiple averaged HDF5 files into a single HDF5 file
    without loading all data into memory.
    """
    total_rows = 0
    feature_count = None
    for path in tmp_paths:
        with h5py.File(path, 'r') as h5_file:
            shape = h5_file['deepprofiler_mean'].shape
            total_rows += shape[0]
            if feature_count is None:
                feature_count = shape[1]

    with h5py.File(output_path, 'w') as h5_out:
        ds_features = h5_out.create_dataset(
            'deepprofiler_mean',
            shape=(total_rows, feature_count),
            dtype='float32'
        )
        ds_wells = h5_out.create_dataset(
            'meta/Metadata_Well',
            shape=(total_rows,),
            dtype=h5py.string_dtype(encoding='utf-8')
        )

        start = 0
        for path in tmp_paths:
            with h5py.File(path, 'r') as h5_file:
                feats = h5_file['deepprofiler_mean'][()]
                wells = h5_file['meta']['Metadata_Well'][()]
                wells = [w.decode('utf-8') if isinstance(w, bytes) else w for w in wells]
                n_rows = feats.shape[0]
                ds_features[start:start+n_rows] = feats
                ds_wells[start:start+n_rows] = wells
                start += n_rows



def main():
    parser = argparse.ArgumentParser(description="Compute well averages or merge averaged HDF5 files.")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="If set, merge multiple averaged HDF5 files instead of computing averages."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input path. For compute: single HDF5 file. For merge: directory or glob pattern of averaged files."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output HDF5 file path."
    )

    args = parser.parse_args()

    if args.merge:
        if os.path.isdir(args.input):
            tmp_paths = sorted(glob.glog(os.path.join(args.input, "*_avg.h5")))
        else:
            tmp_paths = sorted(glob.glob(args.input))
        if not tmp_paths:
            raise FileNotFoundError(f"No files found to merge in {args.input}")
        print(f"Merging {len(tmp_paths)} averaged files into {args.output}")
        merge_avg_h5_stream(tmp_paths, args.output)

    else:
        if not os.path.isfile(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")

        print(f"Computing well averages from {args.input} -> {args.output}")
        compute_well_averages(args.input, args.output)


if __name__ == "__main__":
    main()