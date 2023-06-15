from torch.utils.data import DataLoader, BatchSampler, SequentialSampler

from dataset import AdjacencyCPGDataset


class SizedBatchSampler(BatchSampler):
    def __iter__(self):
        batch = []
        last_size = None
        for idx in self.sampler:
            dim = self.sampler.data_source[idx].shape[1]
            if last_size is not None and (dim != last_size or len(batch) >= self.batch_size) and len(batch) > 0:
                yield batch
                batch = []
            last_size = dim
            batch.append(idx)
        if len(batch) > 0:
            yield batch


def get_adj_dataloader(data_path: str, batch_size: int, model_depth=4):
    dataset = AdjacencyCPGDataset(data_path, model_depth)
    sampler = SizedBatchSampler(SequentialSampler(dataset), batch_size=batch_size, drop_last=False)
    return DataLoader(dataset, batch_sampler=sampler)
