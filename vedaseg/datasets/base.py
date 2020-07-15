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
