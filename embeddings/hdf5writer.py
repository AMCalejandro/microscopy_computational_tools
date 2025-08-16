import numpy as np
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
    
    def add_metadata(self, metadata: dict, group_name="meta"):
        meta_group = self.h5file.create_group(group_name)
        for key, values in metadata.items():
            arr = np.asarray(values)
            if arr.dtype == np.dtype('O'):
                arr = arr.astype('S')
            meta_group.create_dataset(key, data=arr, compression='gzip')

    def add_averages(self, dataset_name: str, filenames: list, group_name="meta_averages"):
        """Compute averages per file_idx and save them as a new dataset + metadata."""
        dset = self.h5file[dataset_name][:]
        file_idx = dset[:, 0].astype(int)
        embeddings = dset[:, 3:]  # skip file_idx, center_i, center_j

        n_files = len(filenames)
        sums = np.zeros((n_files, embeddings.shape[1]), dtype=np.float64)
        counts = np.zeros(n_files, dtype=np.int64)

        for idx, emb in zip(file_idx, embeddings):
            sums[idx] += emb
            counts[idx] += 1

        counts[counts == 0] = 1
        averages = (sums / counts[:, None]).astype('f4')

        self.h5file.create_dataset(
            f"{dataset_name}_averages",
            data=averages,
            compression='gzip'
        )

        self.add_metadata(
            {"file_idx": np.arange(n_files), "filename": filenames},
            group_name=group_name
        )

    def close(self):
        _ = [dset.finalize() for dset in self.datasets]
        self.h5file.close()


class embedding_writer:
    def __init__(self, filename: str, model_name: str, num_rows: int, num_cols: int, dtype='f4'):
        self.h5writer = HDF5Writer(filename)
        self.filenames = []
        num_cols += 3 # add file_idx, center_i, center_j columns
        self.model_name = model_name
        self.dataset = self.h5writer.add_dataset(model_name, num_rows, num_cols, dtype)
    
    def writerows(self, filename, centers_i, centers_j, embeddings):
        if len(self.filenames) == 0 or self.filenames[-1] != filename:
            self.filenames.append(filename)
        file_idx = len(self.filenames) - 1
        meta = np.column_stack([
            np.full(len(centers_i), file_idx, dtype=np.int32),
            centers_i,
            centers_j
        ])
        self.dataset.write_rows(np.concatenate((meta, embeddings), axis=1))

    def close(self, compute_averages=True):
        self.h5writer.add_metadata({"filename": self.filenames}, group_name="meta")
        if compute_averages:
            self.h5writer.add_averages(self.model_name, self.filenames)
        self.h5writer.close()