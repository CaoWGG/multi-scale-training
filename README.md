# multi-scale-training
This is a simple way to implement multi-scale training in pytorch.

### example
```python
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
                                 batch_size=10,
                                 drop_last=True,
                                 multiscale_step=1,
                                 img_sizes=list(range(320, 608 + 1, 32))),
                    num_workers=8)

loader_sequential_sample = DataLoader(dataset=dataset,
                    batch_sampler=BatchSampler(SequentialSampler(dataset),
                                 batch_size=10,
                                 drop_last=True,
                                 multiscale_step=1,
                                 img_sizes=list(range(320, 608 + 1, 32))),
                    num_workers=8)

for batch in loader_random_sample:
    print(batch)
'''random sample
[tensor([ 400, 5006, 9921, 3756, 2826, 6156, 8680, 9827, 4837, 5829]), 
tensor([416, 416, 416, 416, 416, 416, 416, 416, 416, 416])]
[tensor([7319, 4863, 4002, 4321,  838,  736, 9295, 2537, 4451,  492]),
 tensor([352, 352, 352, 352, 352, 352, 352, 352, 352, 352])]
'''
for batch in loader_sequential_sample:
    print(batch)
'''sequential sample
[tensor([8910, 8911, 8912, 8913, 8914, 8915, 8916, 8917, 8918, 8919]), 
tensor([544, 544, 544, 544, 544, 544, 544, 544, 544, 544])]
[tensor([8920, 8921, 8922, 8923, 8924, 8925, 8926, 8927, 8928, 8929]), 
tensor([352, 352, 352, 352, 352, 352, 352, 352, 352, 352])]
'''
```