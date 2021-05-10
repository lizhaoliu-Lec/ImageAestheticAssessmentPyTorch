# Image Aesthetic Assessment in PyTorch

Image Aesthetic Assessment in PyTorch with implemented popular datasets and models (possibly providing the pretrained
ones).

Note that this is a work in process.

## Supported Datasets

- [AVA Datasets](http://refbase.cvc.uab.es/files/MMP2012a.pdf) A large-scale database for aesthetic visual analysis
    - 250,000 images, 235556 (training) and 19926 (testing)
    - aesthetic scores for each image
    - semantic labels for over 60 categories
    - labels related to photographic style
- [AADB Datasets](https://www.ics.uci.edu/~fowlkes/papers/kslmf-eccv16.pdf) A aesthetics and attributes database
    - 10,000 images in total, 8458 (training) and 1,000 (testing)
    - aesthetic scores for each image
    - meaningful attributes assigned to each image (*TODO*)
- [CHAED](https://www.ijcai.org/Proceedings/15/Papers/356.pdf) A chinese handwriting aesthetic evaluation database
    - 1000 Chinese handwriting images 
    - diverse aesthetic qualities for each image (rated by 33 subjects) (*TODO*)

## Supported Models
- [Unified Net](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lee_Image_Aesthetic_Assessment_Based_on_Pairwise_Comparison__A_Unified_ICCV_2019_paper.pdf) Image Aesthetic Assessment Based on Pairwise Comparison – A Unified
Approach to Score Regression, Binary Classification, and Personalization (*In Progress*)
- [PAC-Net](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8451621) PAC-NET: PAIRWISE AESTHETIC COMPARISON NETWORK FOR IMAGE AESTHETIC
ASSESSMENT (*TODO*)
- [NIMA](https://arxiv.org/pdf/1709.05424) NIMA: Neural Image Assessment

## How to use
'''
python -u train.py --config path/to/your/config
'''
