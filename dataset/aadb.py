"""
Photo Aesthetics Ranking Network with Attributes and Content Adaptation. ECCV 2016
A aesthetics and attributes database (AADB) which contains:

===> 0) 10,000 images in total, 8,500 (training), 500 (validation), and 1,000 (testing)
with
===> 1) aesthetic scores
and
===> 2) meaningful attributes assigned to each image
by multiple human rater.

paper reference: https://www.ics.uci.edu/~fowlkes/papers/kslmf-eccv16.pdf
code reference: https://github.com/aimerykong/deepImageAestheticsAnalysis

Training images: https://drive.google.com/uc?export=download&id=1Viswtzb77vqqaaICAQz9iuZ8OEYCu6-_ (2 GB)
Test images: https://drive.google.com/uc?export=download&id=115qnIQ-9pl5Vt06RyFue3b6DabakATmJ (212 MB)
Labels: https://github.com/aimerykong/deepImageAestheticsAnalysis/raw/master/AADBinfo.mat (175 KB)

Preprocess part:
1) unzip AADB_newtest_originalSize.zip
2) unzip datasetImages_originalSize.zip
"""
import scipy.io as scio
from PIL import Image
from torchvision.datasets import VisionDataset

from dataset.utils import check_if_file_exists, join


class AADBBaseDataset(VisionDataset):
    """
    AADBBaseDataset that contains all the elements in AADB datasets,
    which can be inherited into the following datasets:
    1) aesthetic binary classification datasets
    2) aesthetic score regression datasets

    The elements in AVABaseDataset include:
    1) all images
    3) averaged aesthetic scores
    """

    def __init__(self, root, split='train', transforms=None):
        super().__init__(root=root, transforms=transforms)
        assert split in ['train', 'test'], 'Got unsupported split: `%s`' % split
        self.split = split

        if self.split == 'train':
            self.image_root = join(self.root, 'datasetImages_originalSize')
        else:
            self.image_root = join(self.root, 'AADB_newtest_originalSize')

        self.split_and_label_root = join(self.root, 'AADBinfo.mat')

        check_if_file_exists(self.split_and_label_root)

        self._images, self.scores = self._process_split_and_label()

        self._targets = None

    @property
    def images(self):
        return self._images

    @property
    def targets(self):
        return self._targets

    def __getitem__(self, index):
        image_id = self.images[index]
        target = self.targets[index]
        image = Image.open(join(self.image_root, image_id)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.images)

    def _process_split_and_label(self):
        def squeeze_and_to_list(_x):
            return _x.squeeze().tolist()

        def squeeze_and_retrieve_and_to_list(_x):
            tmp = squeeze_and_to_list(_x)
            return [t[0] for t in tmp]

        data = scio.loadmat(self.split_and_label_root)
        for k, v in data.items():
            if k in ['testScore', 'trainScore']:
                data[k] = squeeze_and_to_list(data[k])
            if k in ['testNameList', 'trainNameList']:
                data[k] = squeeze_and_retrieve_and_to_list(data[k])

            # if k in ['testNameList', 'testScore']:
            #     print("===> k: ", k)
            #     print("===> data[k]: ", data[k])

        if self.split == 'train':
            return data['trainNameList'], data['trainScore']
        else:
            return data['testNameList'], data['testScore']


class AADBClassificationDataset(AADBBaseDataset):
    """
    AADBClassificationDataset that used for binary aesthetic classification.
    The binary label is obtained by setting the label to 1 when the aesthetic scores > 0.5
    and setting to 0 otherwise.
    """

    def __init__(self, root, split='train', transforms=None):
        super().__init__(root=root, split=split, transforms=transforms)
        self._targets = [1 if s > 0.5 else 0 for s in self.scores]


class AADBRegressionDataset(AADBBaseDataset):
    """
    AADBRegressionDataset that used for aesthetic score regression,
    since the ranking in aesthetic assessment is often very important.
    """

    def __init__(self, root, split='train', transforms=None):
        super().__init__(root=root, split=split, transforms=transforms)
        self._targets = self.scores


if __name__ == '__main__':
    from tqdm import tqdm
    from random import shuffle
    import matplotlib.pyplot as plt


    def run_process_split_and_label():
        d = AADBBaseDataset(root='/home/liulizhao/datasets/AADB', split='train')


    def run_AADBRegressionDataset():
        d = AADBRegressionDataset(root='/home/liulizhao/datasets/AADB', split='train')
        print(d)

        for image, target in tqdm(d):
            # print(image, target)
            ...


    def run_AADBDataset(AADBDataset, _break=True):
        d_train = AADBDataset(root='/home/liulizhao/datasets/AADB', split='train')
        d_test = AADBDataset(root='/home/liulizhao/datasets/AADB', split='test')

        print("===> Train: \n", d_train)
        print("===> Test: \n", d_test)

        for d in [d_train, d_test]:
            for image, target in tqdm(d):
                # print("===> Image: ", image)
                # print("===> target: ", target)
                if _break:
                    print("===> image: ", image)
                    print("===> target: ", target)
                    break


    def visualize_AADBDataset(AADBDataset, num=3):
        d_train = AADBDataset(root='/home/liulizhao/datasets/AADB', split='train')
        d_test = AADBDataset(root='/home/liulizhao/datasets/AADB', split='test')

        train_ids = list(range(len(d_train)))
        test_ids = list(range(len(d_test)))

        shuffle(train_ids)
        shuffle(test_ids)

        for train_id, test_id in zip(train_ids[:num], test_ids[:num]):
            image, target = d_train[train_id]
            plt.subplot(121)
            plt.axis('off')
            plt.imshow(image)
            plt.title("{0} Train\n Label: {1}".format(AADBDataset.__name__, target))
            plt.tight_layout()
            image, target = d_test[test_id]
            plt.subplot(122)
            plt.axis('off')
            plt.imshow(image)
            plt.title("{0} Test\n Label: {1}".format(AADBDataset.__name__, target))
            plt.tight_layout()
            plt.show()
            plt.close()


    def run_all_dataset():
        run_AADBDataset(AADBClassificationDataset, True)
        run_AADBDataset(AADBRegressionDataset, True)


    def visualize_all_dataset():
        visualize_AADBDataset(AADBClassificationDataset, 3)
        visualize_AADBDataset(AADBRegressionDataset, 3)


    # run_process_split_and_label()
    # run_all_dataset()
    visualize_all_dataset()
