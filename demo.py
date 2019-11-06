from batch_sampler import BatchSampler,RandomSampler,SequentialSampler
from torch.utils.data import Dataset,DataLoader

class my_dataset(Dataset):
    def __init__(self,input_size = 416):
        super(my_dataset,self).__init__()
        self.input_size =  input_size

    def __len__(self):
        return 10000

    def __getitem__(self, item):
        if type(item) == list or type(item) == tuple:
            index,input_size = item
        else:
            index,input_size = item,self.input_size

        return index,input_size


dataset = my_dataset()

loader_random_sample = DataLoader(dataset=dataset,
                    batch_sampler= BatchSampler(RandomSampler(dataset),
                                 batch_size=64,
                                 drop_last=True,
                                 multiscale_step=1,
                                 img_sizes=list(range(320, 608 + 1, 32))),
                    num_workers=8)

loader_sequential_sample = DataLoader(dataset=dataset,
                    batch_sampler=BatchSampler(SequentialSampler(dataset),
                                 batch_size=64,
                                 drop_last=True,
                                 multiscale_step=1,
                                 img_sizes=list(range(320, 608 + 1, 32))),
                    num_workers=8)

for batch in loader_random_sample:
    print(batch)

for batch in loader_sequential_sample:
    print(batch)