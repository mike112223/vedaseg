from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """ BaseDataset
    """
    def __init__(self):
        self.transform = None

    def process(self, image1, mask1, image2=None, mask2=None):
        if self.transform:
            image, mask = self.transform(image1, mask1, image2, mask2)

        return image, mask

    def label_smoothing(self, labels):
        '''Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567.
        inputs: 3d tensor. [N, T, V], where V is the number of vocabulary.
        epsilon: Smoothing rate.
        '''
        labels = labels.float()
        labels[labels == 1] = 1 - self.label_epsilon
        labels[labels == 0] = self.label_epsilon
        return labels
