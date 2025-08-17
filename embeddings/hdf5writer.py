import numpy as np
import pandas as pd
import h5py

class HDF5Dataset:
    """
    Buffered wrapper for writing rows into an HDF5 dataset.
    """
    def __init__(self, dataset: h5py.Dataset):
        self.dataset = dataset
        self.chunk_size = dataset.chunks[0]
        self.rows_written = 0
        self.buffer = []
        self.buffer_len = 0

    def _write(self, data: np.ndarray):
        num_rows = data.shape[0]
        self.dataset[self.rows_written:self.rows_written + num_rows, :] = data
        self.rows_written += num_rows

    def write_rows(self, data: np.ndarray):
        self.buffer.append(data)
        self.buffer_len += data.shape[0]
        if self.buffer_len < self.chunk_size:
            return
        buffer = np.concatenate(self.buffer, axis=0)
        self.buffer = []
        self.buffer_len -= self.chunk_size
        if buffer.shape[0] > self.chunk_size:
            self.buffer.append(buffer[self.chunk_size:, :])
            buffer = buffer[:self.chunk_size, :]
        self._write(buffer)

    def finalize(self):
        if self.buffer_len > 0:
            buffer = np.concatenate(self.buffer, axis=0)
            self._write(buffer)


class HDF5Writer:
    """
    Wrapper for creating and writing to an HDF5 file.
    """
    def __init__(self, filename: str):
        self.h5file = h5py.File(filename, 'w')
        self.datasets = []

    def add_dataset(self, name: str, num_rows: int, num_cols: int, dtype='f4') -> HDF5Dataset:
        dataset = self.h5file.create_dataset(
            name, shape=(num_rows, num_cols), maxshape=(num_rows, num_cols),
            dtype=dtype, compression='gzip'
        )
        dataset = HDF5Dataset(dataset)
        self.datasets.append(dataset)
        return dataset
    
    def add_metadata(self, metadata: pd.DataFrame):
        meta_group = self.h5file.create_group("meta")
        for col in metadata.columns:
            col_data = metadata[col].values
            if metadata[col].dtype == np.dtype('O'):
                col_data = col_data.astype('S')
            meta_group.create_dataset(col, data=col_data, compression='gzip')

    def compute_averages(self, dataset_name: str):
        """Compute averages of embeddings per file_idx, add as <dataset>_averages."""
        dset = self.h5file[dataset_name]
        data = dset[:]  # load all
        file_idx = data[:, 0].astype(int)
        embeddings = data[:, 3:]  # skip first 3 metadata columns

        averages = []
        indices = []
        for idx in np.unique(file_idx):
            mask = (file_idx == idx)
            avg_vec = embeddings[mask].mean(axis=0)
            averages.append(avg_vec)
            indices.append(idx)

        averages = np.vstack(averages)
        indices = np.array(indices).reshape(-1, 1)

        result = np.hstack((indices, averages))

        avg_dset = self.h5file.create_dataset(
            f"{dataset_name}_averages",
            shape=result.shape,
            dtype='f4',
            compression='gzip'
        )
        avg_dset[:, :] = result

    def close(self):
        _ = [dset.finalize() for dset in self.datasets]
        self.h5file.close()


class embedding_writer:
    def __init__(self, filename: str, model_name: str, num_rows: int, num_cols: int, dtype='f4'):
        self.h5writer = HDF5Writer(filename)
        self.model_name = model_name
        self.filenames = []
        num_cols += 3 # add file_idx, center_i, center_j columns
        self.dataset = self.h5writer.add_dataset(model_name, num_rows, num_cols, dtype)
    
    def writerows(self, filename, centers_i, centers_j, embeddings):
        if len(self.filenames) == 0 or self.filenames[-1] != filename:
            self.filenames.append(filename)
        file_idx = len(self.filenames) - 1
        meta = np.array([[file_idx]*len(centers_i), centers_i, centers_j]).T
        self.dataset.write_rows(np.concatenate((meta, embeddings), axis=1))

    def close(self, compute_averages = True):
        meta = pd.DataFrame({'filename': self.filenames})
        self.h5writer.add_metadata(meta)
        if compute_averages:
            self.h5writer.compute_averages(self.model_name)
        self.h5writer.close()
