"""
CHAED includes
1) 1000 images, 10 for each of 100 Chinese characters
2) including 20 characters with single element and
3) 80 with multi components in different structure types including horizontal composition, vertical composition,
                                                            half surrounding composition and surrounding composition.
We invited users to
4) evaluate each image's visual quality by 3 levels, good, medium and bad
through the website http://59.108.48.27/eval.
In "Aesthetic Visual Quality Evaluation of Chinese Handwritings", for every character,
5) the odd number of images are used to train the ANN and
6) the even number of images are used to test it.
That's to say, 001.jpg, 003.jpg, 005.jpg, 007.jpg, 009.jpg are the training set and
002.jpg, 004.jpg, 006.jpg, 008.jpg, 010.jpg are the test set.
"""
import os
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset

from common.utils import join

from dataset import DatasetFactory


class CHAEDBaseDataset(VisionDataset):
    """
    CHAEDDataset that contains all the elements in CHAED datasets,
    which can be inherited into the following datasets:
    1) aesthetic ternary classification dataset
    2) aesthetic score regression dataset
    3) aesthetic score distribution dataset

    The elements in AVABaseDataset include:
    1) all images
    3) visual quality scores from 3 levels, good, medium and bad
    4) visual quality score range from 0 to 100

    Generally, the probabilities of good, medium and bad evaluation results of each image
    are denoted as pgood, pmedium and pbad.

    The aesthetic score is defined by S = 100 × pgood + 50 × pmedium + 0 × pbad

    """

    def __init__(self, root, split='train', transforms=None):
        super().__init__(root=root, transforms=transforms)
        assert split in ['train', 'test'], 'Got unsupported split: `%s`' % split
        self.split = split

        self.characters_txt_path = join(self.root, 'Characters.txt')
        self.evaluation_data_txt_path = join(self.root, 'EvaluationData.txt')
        self.score_analysis_per_image_txt_path = join(self.root, 'ScoreAnalysisPerImage.txt')

        self.characters = self.read_characters_txt()
        self.charID_to_scores = self.read_evaluation_data_txt()
        self.charID_to_tuple = self.read_score_analysis_txt()
        self.charID_to_img_path = {k: join(self.root, v[0], v[1]) for k, v in self.charID_to_tuple.items()}
        self.charID_to_avg_score = {k: v[2] for k, v in self.charID_to_tuple.items()}
        self.charID_to_sigma = {k: v[3] for k, v in self.charID_to_tuple.items()}
        self.charID_to_distribution, self.charID_to_ternary_label = self.get_distribution_and_ternary()

        self.remove_missing_character()

        self.split_charID = self.get_charID_according_to_split()

        self._images = {k: self.charID_to_img_path[k] for k in self.split_charID}

        self._targets = None

    @property
    def images(self):
        return self._images

    @property
    def targets(self):
        return self._targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        charID = self.split_charID[index]
        image_path = self.images[charID]
        target = self.targets[charID]
        image = Image.open(image_path).convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def get_distribution_and_ternary(self):
        charID_to_distribution = {}
        charID_to_ternary_label = {}

        for k, v in self.charID_to_scores.items():
            p0 = list.count(v, 0) / len(v)
            p50 = list.count(v, 50) / len(v)
            p100 = list.count(v, 100) / len(v)
            charID_to_distribution[k] = [p0, p50, p100]
            charID_to_ternary_label[k] = np.argmax(charID_to_distribution[k])

        return charID_to_distribution, charID_to_ternary_label

    def read_characters_txt(self):
        with open(self.characters_txt_path, encoding='gbk') as f:
            lines = f.readlines()
            # strip line break
            lines = [line.strip('\n') for line in lines]
            # get characters
            characters = [character for line in lines for character in line]
            return characters

    def read_evaluation_data_txt(self):
        with open(self.evaluation_data_txt_path) as f:
            lines = f.readlines()
            lines = [line.strip('\n') for line in lines]
            # remove 'charID	score'
            lines = lines[1:]
            charID_to_scores = {}
            # retrieve 33 score for each char (some chars may only have 30, 32, 34 scores)
            for line in lines:
                charID, score = line.split('\t')
                score = int(score)
                if charID not in charID_to_scores:
                    charID_to_scores[charID] = []
                charID_to_scores[charID].append(score)
        return charID_to_scores

    def read_score_analysis_txt(self):
        with open(self.score_analysis_per_image_txt_path) as f:
            lines = f.readlines()
            lines = [line.strip('\n') for line in lines]
            # remove 'charName	imageID	charID	average score	sigma'
            lines = lines[1:]
            charID_to_tuple = {}

            for line in lines:
                charName, imageID, charID, avg_score, sigma = line.split('\t')
                avg_score = float(avg_score)
                sigma = float(sigma)
                charID_to_tuple[charID] = (charName, imageID, avg_score, sigma)

        return charID_to_tuple

    def remove_missing_character(self):
        # character GB3632 is missing
        # correspond to key id range from 231 to 240
        for i in range(231, 240 + 1):
            self.charID_to_scores.pop(str(i))
            self.charID_to_tuple.pop(str(i))
            self.charID_to_img_path.pop(str(i))
            self.charID_to_avg_score.pop(str(i))
            self.charID_to_sigma.pop(str(i))

    def get_charID_according_to_split(self):
        split = self.split
        split_charID = []
        for k, v in self.charID_to_img_path.items():
            filename = os.path.basename(v)
            file_number = int(filename.split('.')[0])
            # if split == 'train' and int():
            #     ...
            if file_number % 2 == 0 and split == 'test':
                split_charID.append(k)
            if file_number % 2 != 0 and split == 'train':
                split_charID.append(k)
        return split_charID


@DatasetFactory.register('CHAEDClassificationDataset')
class CHAEDClassificationDataset(CHAEDBaseDataset):
    """
    CHAED Classification Dataset for ternary classification.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._targets = self.charID_to_ternary_label


@DatasetFactory.register('CHAEDDistributionDataset')
class CHAEDDistributionDataset(CHAEDBaseDataset):
    """
    CHAED Classification Dataset for ternary score distribution matching.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._targets = self.charID_to_distribution


@DatasetFactory.register('CHAERegressionDataset')
class CHAERegressionDataset(CHAEDBaseDataset):
    """
    CHAED Classification Dataset for score regression.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._targets = self.charID_to_avg_score


if __name__ == '__main__':
    def run_chaed():
        d = CHAEDBaseDataset(root='/home/liulizhao/datasets/CHAED', split='test')
        print("===> d ", d)


    def run_CHAEDDataset(CHAEDDataset, _break=True):
        from tqdm import tqdm
        d_train = CHAEDDataset(root='/home/liulizhao/datasets/CHAED', split='train')
        d_test = CHAEDDataset(root='/home/liulizhao/datasets/CHAED', split='test')

        print("===> Train: \n", d_train)
        print("===> Test: \n", d_test)

        for d in [d_train, d_test]:
            for image, target in tqdm(d):
                # print("===> Image: ", image)
                # print("===> target: ", target)
                if _break:
                    print("===> image: ", image.size)
                    print("===> target: ", target)
                    # break


    def run_all_dataset():
        break_or_not = True
        run_CHAEDDataset(CHAEDClassificationDataset, break_or_not)
        run_CHAEDDataset(CHAEDDistributionDataset, break_or_not)
        run_CHAEDDataset(CHAERegressionDataset, break_or_not)


    def visualize_CHAEDDataset(CHAEDDataset, num=3):
        from random import shuffle
        import matplotlib.pyplot as plt
        d_train = CHAEDDataset(root='/home/liulizhao/datasets/CHAED', split='train')
        d_test = CHAEDDataset(root='/home/liulizhao/datasets/CHAED', split='test')

        train_ids = list(range(len(d_train)))
        test_ids = list(range(len(d_test)))

        shuffle(train_ids)
        shuffle(test_ids)

        for train_id, test_id in zip(train_ids[:num], test_ids[:num]):
            image, target = d_train[train_id]
            if isinstance(target, list):
                target = [round(t, 2) for t in target]
            else:
                target = round(target, 2)
            plt.subplot(121)
            plt.axis('off')
            plt.imshow(image)
            plt.title("{0} Train\n Label: {1}".format(CHAEDDataset.__name__, target))
            plt.tight_layout()
            image, target = d_test[test_id]
            if isinstance(target, list):
                target = [round(t, 2) for t in target]
            else:
                target = round(target, 2)
            plt.subplot(121)
            plt.subplot(122)
            plt.axis('off')
            plt.imshow(image)
            plt.title("{0} Test\n Label: {1}".format(CHAEDDataset.__name__, target))
            plt.tight_layout()
            plt.show()
            plt.close()


    def visualize_all_dataset():
        visualize_CHAEDDataset(CHAEDClassificationDataset, 3)
        visualize_CHAEDDataset(CHAEDDistributionDataset, 3)
        visualize_CHAEDDataset(CHAERegressionDataset, 3)


    # run_chaed()
    run_all_dataset()
    # visualize_all_dataset()
