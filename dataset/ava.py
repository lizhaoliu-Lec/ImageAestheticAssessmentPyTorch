"""
AVA: A large-scale database for aesthetic visual analysis. CVPR 2012
A large-scale database for conducting Aesthetic Visual Analysis: AVA.

It contains:
===> 1)　over 250,000 images, 235,599 (training) and 19,930 (testing)
along with a rich variety of meta-data including
===> 2) a large number of aesthetic scores for each image,
===> 3) semantic labels for over 60 categories as well as
===> 4) labels related to photographic style.

paper reference: http://refbase.cvc.uab.es/files/MMP2012a.pdf
code reference: https://github.com/mtobeiyf/ava_downloader

You can download the whole dataset using the following url with torrent
whole dataset: http://academictorrents.com/download/71631f83b11d3d79d8f84efe0a7e12f0ac001460.torrent (~30 GB)
or, you can refer to the code reference mentioned above and download it via its instructions.

Preprocess part:
1) cd images
2) 7z x images.7z.001
"""
import os
from PIL import Image

from torchvision.datasets import VisionDataset

from dataset.utils import check_if_file_exists, join


class AVABaseDataset(VisionDataset):
    """
    AVABaseDataset that contains all the elements in AVA datasets,
    which can be inherited into the following datasets:
    1) aesthetic binary classification datasets
    2) aesthetic score regression datasets
    3) style classification datasets

    The elements in AVABaseDataset include:
    1) all images
    2) aesthetic scores count
    3) averaged aesthetic scores
    4) aesthetic scores count distribution
    5) aesthetic classification scores
    6) style categories
    7) challenge categories
    8) content categories #TODO for now not supported
    """

    CORRUPTED_IMAGE_IDS = [
        '512522', '627334', '523555', '593733',
        '230701', '564093', '501015', '371434',
        '440774', '501064', '453704', '2129',
        '179118', '1617', '476416', '570175',
        '547917', '502377', '499068', '564307',
        '729377', '639811', '532055', '501095',
        '277832', '556798',
    ]

    def __init__(self, root, split='train', transforms=None):
        super().__init__(root=root, transforms=transforms)
        assert split in ['train', 'test'], 'Got unsupported split: `%s`' % split
        self.split = split

        self.image_root = join(self.root, 'images/images')
        self.aesthetics_split_path = join(self.root, 'aesthetics_image_lists')
        self.style_split_path = join(self.root, 'style_image_lists')
        self.ava_txt_path = join(self.root, 'AVA.txt')
        self.tags_txt_path = join(self.root, 'tags.txt')
        self.challenges_txt_path = join(self.root, 'challenges.txt')
        self.style_txt_path = join(self.root, 'style_image_lists/styles.txt')
        self.style_content_path = join(self.root, 'style_image_lists')

        check_if_file_exists(self.aesthetics_split_path)
        check_if_file_exists(self.style_split_path)
        check_if_file_exists(self.ava_txt_path)
        check_if_file_exists(self.tags_txt_path)
        check_if_file_exists(self.challenges_txt_path)
        check_if_file_exists(self.style_txt_path)
        check_if_file_exists(self.style_content_path)

        self.aesthetics_test_split = self._process_aesthetics_split('test')
        self.aesthetics_train_split = self._process_aesthetics_split('train')

        self.style_split = self._process_style_split()
        self.imageId2ava_contents, all_aesthetics_imageIds = self._process_ava_txt()
        self.imageId2style = self._process_style_content()
        self.semanticId2name = self._process_id2name_txt(self.tags_txt_path)
        self.challengeId2name = self._process_id2name_txt(self.challenges_txt_path)
        self.style2name = self._process_id2name_txt(self.style_txt_path)

        self.aesthetics_split = self._post_process_aesthetics_split(self.split,
                                                                    self.aesthetics_test_split,
                                                                    all_aesthetics_imageIds)

        self._images = None
        self._targets = None
        self.all_imageIds = self._get_all_image_list()
        self.aesthetics_split = self._post_process_image_list(self.aesthetics_split, self.all_imageIds,
                                                              self.CORRUPTED_IMAGE_IDS)
        self.style_split = self._post_process_image_list(self.style_split, self.all_imageIds, self.CORRUPTED_IMAGE_IDS)

    @property
    def images(self):
        return self._images

    @property
    def targets(self):
        return self._targets

    def __getitem__(self, index):
        image_id = self.images[index]
        image = Image.open(join(self.image_root, image_id + '.jpg')).convert('RGB')
        target = self.targets[image_id]

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.images)

    def _process_aesthetics_split(self, split):
        p = self.aesthetics_split_path

        split_files = os.listdir(p)
        # print("1 ===> split_files (%d): " % len(split_files), split_files)

        # remove the small scale train split
        split_files.remove('generic_ss_train.jpgl')
        # print("2 ===> split_files (%d): " % len(split_files), split_files)

        # acquire the desired split e.g., remove other split
        split_files = [f for f in split_files if split in f]
        # print("3 ===> split_files (%d): " % len(split_files), split_files)

        # get the full path
        split_files = [join(p, f) for f in split_files]
        # print("4 ===> split_files (%d): " % len(split_files), split_files)

        aesthetics_split_image_ids = []
        s = 0
        for f in split_files:
            with open(f) as _f:
                tmp = [_id.strip() for _id in _f.readlines()]
                s += len(tmp)
                # print("===> %s: %d" % (str(f), len(tmp)))
                aesthetics_split_image_ids.extend(tmp)
        # print(aesthetics_split_image_ids)
        # print("===> len: ", len(aesthetics_split_image_ids))
        # print("===> len(set): ", len(set(aesthetics_split_image_ids)))
        # print("===> s: %d" % s)
        return list(set(aesthetics_split_image_ids))

    @staticmethod
    def _post_process_aesthetics_split(split, test_split, all_aesthetics_imageIds):
        # TODO: with image exists check
        if split == 'test':
            return list(set(test_split) & set(all_aesthetics_imageIds))
        else:
            return list(set(all_aesthetics_imageIds) - set(test_split))

    def _process_style_split(self):
        f = join(self.style_split_path, '%s.jpgl' % self.split)
        style_split_image_ids = []
        with open(f) as _f:
            style_split_image_ids.extend([_.replace('\n', '') for _ in _f.readlines()])
        # print("===> len(style_split_image_ids)-%s" % self.split, len(style_split_image_ids))
        return style_split_image_ids

    def _process_ava_txt(self):
        # each line like this:
        # 1 953619 0 1 5 17 38 36 15 6 5 1 1 22 1396
        # index, image id, score_cnt x 10, semantic_id x2, challenge_id
        imageId2ava_contents = {}
        imageIds = []
        with open(self.ava_txt_path) as _f:
            for line in _f.readlines():
                line = line.split(' ')
                assert len(line) == 15, 'Corrupted AVA.txt'
                # index = line[0]
                image_id = line[1]
                counts = line[2:11]
                semantic_ids = line[12:14]
                challenge_id = line[14]
                imageIds.append(image_id)
                imageId2ava_contents[image_id] = {
                    'image_id': image_id,
                    'counts': [int(_) for _ in counts],
                    'semantic_ids': [int(_) for _ in semantic_ids],
                    'challenge_id': int(challenge_id),
                }
                counts = imageId2ava_contents[image_id]['counts']
                # must start from 1, because rating is from 1 to 10
                sum_counts = sum(counts)
                sum_scores = sum([i * _ for i, _ in enumerate(counts, start=1)])
                imageId2ava_contents[image_id]['count_distribution'] = [_ / sum_counts for _ in counts]
                regression_score = sum_scores / sum_counts
                imageId2ava_contents[image_id]['regression_score'] = regression_score
                imageId2ava_contents[image_id]['classification_score'] = 1 if regression_score > 5 else 0

        return imageId2ava_contents, imageIds

    def _process_style_content(self):
        split = self.split
        filename = 'train.lab' if split == 'train' else 'test.multilab'
        f = join(self.style_content_path, filename)
        imageId2style = {}
        style_list = []
        with open(f) as _f:
            if split == 'train':
                style_list = _f.readlines()
                style_list = [int(_.replace('\n', '')) for _ in style_list]
            else:
                for line in _f.readlines():
                    line = line.replace('\n', '').split(' ')
                    style_list.append([i for i, v in enumerate(line, 1) if v == '1'])

        for image_id, style in zip(self.style_split, style_list):
            imageId2style[image_id] = style

        return imageId2style

    def _get_all_image_list(self):
        return [_.split('.')[0] for _ in os.listdir(self.image_root)]

    @staticmethod
    def _process_id2name_txt(path):
        id2name = {}
        with open(path) as _f:
            for line in _f.readlines():
                line = line.strip().replace('\n', '')
                line = line.split(' ', 1)
                assert len(line) == 2, 'Corrupted file: `%s`' % path
                _id, name = line[0], line[1]
                id2name[_id] = name

        return id2name

    @staticmethod
    def _post_process_image_list(image_list, all_image_list, CORRUPTED_IMAGE_IDS):
        return list(set(image_list) & set(all_image_list) - set(CORRUPTED_IMAGE_IDS))


class AVAAestheticClassificationDataset(AVABaseDataset):
    """
    AVAAestheticClassificationDataset that used for binary aesthetic classification.
    The binary label is obtained by setting the label to 1 when the averaged aesthetic scores > 5
    and setting to 0 otherwise.
    """

    def __init__(self, root, split='train', transforms=None):
        super().__init__(root=root, split=split, transforms=transforms)
        self._images = self.aesthetics_split
        self._targets = {k: self.imageId2ava_contents[k]['classification_score'] for k in
                         self.imageId2ava_contents.keys()}


class AVAAestheticDistributionDataset(AVABaseDataset):
    """
    AVAAestheticDistributionDataset that used for aesthetic distribution matching,
    since the binary classification is often too hard and not accurate enough to reflect the image level aesthetic.
    The aesthetic distribution is obtained by normalizing the score count from 1 to 10.
    """

    def __init__(self, root, split='train', transforms=None):
        super().__init__(root=root, split=split, transforms=transforms)
        self._images = self.aesthetics_split
        self._targets = {k: self.imageId2ava_contents[k]['count_distribution'] for k in
                         self.imageId2ava_contents.keys()}


class AVAAestheticRegressionDataset(AVABaseDataset):
    """
    AVAAestheticRegressionDataset that used for aesthetic score regression,
    since the ranking in aesthetic assessment is often very important.
    The aesthetic score is obtained by averaging the score from 1 to 10.
    """

    def __init__(self, root, split='train', transforms=None):
        super().__init__(root=root, split=split, transforms=transforms)
        self._images = self.aesthetics_split
        self._targets = {k: self.imageId2ava_contents[k]['regression_score'] for k in
                         self.imageId2ava_contents.keys()}


class AVAStyleClassificationDataset(AVABaseDataset):
    def __init__(self, root, split='train', transforms=None):
        super().__init__(root=root, split=split, transforms=transforms)
        self._images = self.style_split
        self._targets = self.imageId2style


if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from random import shuffle


    def print_dict_with_limit(d: dict, limit: int, name: str):
        cnt = 0
        print("===> %s len: " % name, len(d))
        for k, v in d.items():
            print("===> key -:- v ", k, ' -:- ', v)
            cnt += 1
            if cnt > limit:
                break


    def detect_invalid_char(s):
        for char in s:
            if char not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                print("===> `%s` contains invalid char" % s)


    def run_process_aesthetics_split():
        d_train = AVABaseDataset(root='E:/Datasets/AVA', split='train')
        d_test = AVABaseDataset(root='E:/Datasets/AVA', split='test')
        print("===> len(train): ", len(d_train.aesthetics_split))
        print("===> len(test): ", len(d_test.aesthetics_split))
        join_split = d_train.aesthetics_split + d_test.aesthetics_split
        print("===> join length: ", len(join_split))
        print("===> join set length: ", len(set(join_split)))
        print("===> intersect length: ", len(set(d_train.aesthetics_split).intersection(set(d_test.aesthetics_split))))


    def run_process_ava_txt():
        d_train = AVABaseDataset(root='E:/Datasets/AVA', split='train')
        print("===> len(image_id2contents): ", len(d_train.imageId2ava_contents))
        print_dict_with_limit(d_train.imageId2ava_contents, 3, 'image_id2contents')


    def run_process_id2name_txt():
        d_train = AVABaseDataset(root='E:/Datasets/AVA', split='train')
        semantic = d_train.semanticId2name
        challenge = d_train.challengeId2name
        style = d_train.style2name
        print_dict_with_limit(semantic, 2, 'semantic')
        print_dict_with_limit(challenge, 2, 'challenge')
        print_dict_with_limit(style, 2, 'style')


    def run_process_style_split():
        d_train = AVABaseDataset(root='E:/Datasets/AVA', split='train')
        d_test = AVABaseDataset(root='E:/Datasets/AVA', split='test')
        join_split = d_train.style_split + d_test.style_split
        print("===> join length: ", len(join_split))
        print("===> join set length: ", len(set(join_split)))
        print("===> intersect length: ", len(set(d_train.aesthetics_split).intersection(set(d_test.aesthetics_split))))


    def run_process_style_content():
        d_train = AVABaseDataset(root='E:/Datasets/AVA', split='train')
        print("===> len(imageId2style): ", len(d_train.imageId2style))
        print_dict_with_limit(d_train.imageId2style, 100, 'imageId2style')


    def compare_ava_and_aesthetic_split():

        d_train = AVABaseDataset(root='E:/Datasets/AVA', split='train')
        d_test = AVABaseDataset(root='E:/Datasets/AVA', split='test')
        # aesthetics_split = d_train.aesthetics_split + d_test.aesthetics_split
        aesthetics_split = d_test.aesthetics_split

        # find out the split not in ava
        cnt = 0

        for image_id in tqdm(aesthetics_split):
            if image_id not in d_train.imageId2ava_contents:
                print("%s not in ava" % image_id)
                cnt += 1
        print("The num of not in ava: %d" % cnt)

        print("===> Detecting invalid char in ava...")
        for k in d_train.imageId2ava_contents.keys():
            detect_invalid_char(k)

        print("===> Detecting invalid char in train_split")
        for s in d_train.aesthetics_split:
            detect_invalid_char(s)

        print("===> Detecting invalid char in test_split")
        for s in d_test.aesthetics_split:
            detect_invalid_char(s)


    def run_AVADataset(AVADataset, _break=True):
        d_train = AVADataset(root='/home/liulizhao/datasets/AVA_dataset', split='train')
        d_test = AVADataset(root='/home/liulizhao/datasets/AVA_dataset', split='test')

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


    def run_all_dataset():
        run_AVADataset(AVAAestheticClassificationDataset, False)
        run_AVADataset(AVAAestheticDistributionDataset)
        run_AVADataset(AVAAestheticRegressionDataset)
        run_AVADataset(AVAStyleClassificationDataset, False)


    def visualize_AVADataset(AVADataset, num=3):
        d_train = AVADataset(root='/home/liulizhao/datasets/AVA_dataset', split='train')
        d_test = AVADataset(root='/home/liulizhao/datasets/AVA_dataset', split='test')

        train_ids = list(range(len(d_train)))
        test_ids = list(range(len(d_test)))

        shuffle(train_ids)
        shuffle(test_ids)

        for train_id, test_id in zip(train_ids[:num], test_ids[:num]):
            image, target = d_train[train_id]
            plt.subplot(121)
            plt.axis('off')
            plt.imshow(image)
            if isinstance(target, list):
                target = ['%.2f' % t for t in target]
            plt.title("{0} Train\n Label: {1}".format(AVADataset.__name__, target))
            plt.tight_layout()
            image, target = d_test[test_id]
            if isinstance(target, list):
                target = ['%.2f' % t for t in target]
            plt.subplot(122)
            plt.axis('off')
            plt.imshow(image)
            plt.title("{0} Test\n Label: {1}".format(AVADataset.__name__, target))
            plt.tight_layout()
            plt.show()
            plt.close()


    def visualize_all_dataset():
        visualize_AVADataset(AVAAestheticClassificationDataset)
        visualize_AVADataset(AVAAestheticDistributionDataset)
        visualize_AVADataset(AVAAestheticRegressionDataset)
        visualize_AVADataset(AVAStyleClassificationDataset)


    # run_process_aesthetics_split()
    # run_process_ava_txt()
    # run_process_id2name_txt()
    # run_process_style_split()
    # run_process_style_content()
    # compare_ava_and_aesthetic_split()
    # run_all_dataset()
    visualize_all_dataset()
