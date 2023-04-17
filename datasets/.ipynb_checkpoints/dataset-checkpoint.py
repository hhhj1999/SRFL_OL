import numpy as np
import imageio
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import json
from os.path import join

from torchvision.datasets.utils import list_dir, download_url
import scipy
from scipy import io
import scipy.misc
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

class CUB():
    def __init__(self, input_size, root, is_train=True, data_len=None):
        self.input_size = input_size
        self.root = root
        self.is_train = is_train
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        box_file = open(os.path.join(self.root, 'bounding_boxes.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        box_file_list = []
        for line in box_file:
            data = line[:-1].split(' ')
            box_file_list.append([int(float(data[2])), int(float(data[1])),
                                  int(float(data[4])), int(float(data[3]))])
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
        self.train_box = torch.tensor([x for i, x in zip(train_test_list, box_file_list) if i])
        self.test_box = torch.tensor([x for i, x in zip(train_test_list, box_file_list) if not i])
        if self.is_train:
            self.train_img = [os.path.join(self.root, 'images', train_file) for train_file in
                              train_file_list[:data_len]]
            # self.train_img = [imageio.imread(os.path.join(self.root, 'images', train_file)) for train_file in
            #                   train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        if not self.is_train:
            self.test_img = [os.path.join(self.root, 'images', test_file) for test_file in
                             test_file_list[:data_len]]
            # self.test_img = [imageio.imread(os.path.join(self.root, 'images', test_file)) for test_file in
            #                  test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target, box = imageio.imread(self.train_img[index]), self.train_label[index], self.train_box[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            # compute scaling
            height, width = img.height, img.width
            height_scale = self.input_size / height
            width_scale = self.input_size / width

            img = transforms.Resize((510, 510), Image.BILINEAR)(img)
            img = transforms.RandomCrop((448, 448))(img)
            img = transforms.RandomHorizontalFlip()(img)
#             img = transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1)(img)
#             img = transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1)(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target, box = imageio.imread(self.test_img[index]), self.test_label[index], self.test_box[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

            # compute scaling
            height, width = img.height, img.width
            height_scale = self.input_size / height
            width_scale = self.input_size / width

            img = transforms.Resize((510, 510), Image.BILINEAR)(img)
            img = transforms.CenterCrop((448, 448))(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        scale = torch.tensor([height_scale, width_scale])

        return img, target, box, scale

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)

class STANFORD_CAR():
    def __init__(self, input_size, root, is_train=True, data_len=None):
        self.input_size = input_size
        self.root = root
        self.is_train = is_train
        train_img_path = os.path.join(self.root, 'cars_train')
        test_img_path = os.path.join(self.root, 'cars_test')
        train_label_file = open(os.path.join(self.root, 'train.txt'))
        test_label_file = open(os.path.join(self.root, 'test.txt'))
        train_img_label = []
        test_img_label = []
        for line in train_label_file:
            train_img_label.append([os.path.join(train_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        for line in test_label_file:
            test_img_label.append([os.path.join(test_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        self.train_img_label = train_img_label[:data_len]
        self.test_img_label = test_img_label[:data_len]


    def __getitem__(self, index):
        if self.is_train:
            img, target = imageio.imread(self.train_img_label[index][0]), self.train_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

#             img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
#             # img = transforms.RandomResizedCrop(size=self.input_size,scale=(0.4, 0.75),ratio=(0.5,1.5))(img)
#             # img = transforms.RandomCrop(self.input_size)(img)
#             img = transforms.RandomHorizontalFlip()(img)
#             img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

#             img = transforms.ToTensor()(img)
#             img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            
            
            img = transforms.Resize((510, 510), Image.BILINEAR)(img)
            img = transforms.RandomCrop((448, 448))(img)
            img = transforms.RandomHorizontalFlip()(img)
#             img = transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1)(img)
#             img = transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1)(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = imageio.imread(self.test_img_label[index][0]), self.test_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            
#             img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
#             # img = transforms.CenterCrop(self.input_size)(img)
#             img = transforms.ToTensor()(img)
#             img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

            img = transforms.Resize((510, 510), Image.BILINEAR)(img)
            img = transforms.CenterCrop((448, 448))(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)


        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_img_label)
        else:
            return len(self.test_img_label)

class FGVC_aircraft():
    def __init__(self, input_size, root, is_train=True, data_len=None):
        self.input_size = input_size
        self.root = root
        self.is_train = is_train
        train_img_path = os.path.join(self.root, 'data', 'images')
        test_img_path = os.path.join(self.root, 'data', 'images')
        train_label_file = open(os.path.join(self.root, 'data', 'train.txt'))
        test_label_file = open(os.path.join(self.root, 'data', 'test.txt'))
        train_img_label = []
        test_img_label = []
        for line in train_label_file:
            train_img_label.append([os.path.join(train_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        for line in test_label_file:
            test_img_label.append([os.path.join(test_img_path,line[:-1].split(' ')[0]), int(line[:-1].split(' ')[1])-1])
        self.train_img_label = train_img_label[:data_len]
        self.test_img_label = test_img_label[:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = imageio.imread(self.train_img_label[index][0]), self.train_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')

#             img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
#             # img = transforms.RandomResizedCrop(size=self.input_size,scale=(0.4, 0.75),ratio=(0.5,1.5))(img)
#             # img = transforms.RandomCrop(self.input_size)(img)
#             img = transforms.RandomHorizontalFlip()(img)
#             img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

#             img = transforms.ToTensor()(img)
#             img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            
            
            img = transforms.Resize((510, 510), Image.BILINEAR)(img)
            img = transforms.RandomCrop((448, 448))(img)
            img = transforms.RandomHorizontalFlip()(img)
#             img = transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1)(img)
#             img = transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1)(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)

            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = imageio.imread(self.test_img_label[index][0]), self.test_img_label[index][1]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
#             img = transforms.Resize((self.input_size, self.input_size), Image.BILINEAR)(img)
#             # img = transforms.CenterCrop(self.input_size)(img)
#             img = transforms.ToTensor()(img)
#             img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            img = transforms.Resize((510, 510), Image.BILINEAR)(img)
            img = transforms.CenterCrop((448, 448))(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_img_label)
        else:
            return len(self.test_img_label)


class dogs(Dataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    folder = 'dog'
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

    def __init__(self,
                 root,
                 train=True,
                 cropped=False,
                 transform=None,
                 target_transform=None,
                 download=False):

        # self.root = join(os.path.expanduser(root), self.folder)
        self.root = root
        self.train = train
        self.cropped = cropped
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        split = self.load_split()

        self.images_folder = join(self.root, 'Images/Images')
        self.annotations_folder = join(self.root, 'Annotation/Annotation')
        self._breeds = list_dir(self.images_folder)

        if self.cropped:
            self._breed_annotations = [[(annotation, box, idx)
                                        for box in self.get_boxes(join(self.annotations_folder, annotation))]
                                       for annotation, idx in split]
            self._flat_breed_annotations = sum(self._breed_annotations, [])

            self._flat_breed_images = [(annotation + '.jpg', idx) for annotation, box, idx in
                                       self._flat_breed_annotations]
        else:
            self._breed_images = [(annotation + '.jpg', idx) for annotation, idx in split]

            self._flat_breed_images = self._breed_images

        self.classes = ["Chihuaha",
                        "Japanese Spaniel",
                        "Maltese Dog",
                        "Pekinese",
                        "Shih-Tzu",
                        "Blenheim Spaniel",
                        "Papillon",
                        "Toy Terrier",
                        "Rhodesian Ridgeback",
                        "Afghan Hound",
                        "Basset Hound",
                        "Beagle",
                        "Bloodhound",
                        "Bluetick",
                        "Black-and-tan Coonhound",
                        "Walker Hound",
                        "English Foxhound",
                        "Redbone",
                        "Borzoi",
                        "Irish Wolfhound",
                        "Italian Greyhound",
                        "Whippet",
                        "Ibizian Hound",
                        "Norwegian Elkhound",
                        "Otterhound",
                        "Saluki",
                        "Scottish Deerhound",
                        "Weimaraner",
                        "Staffordshire Bullterrier",
                        "American Staffordshire Terrier",
                        "Bedlington Terrier",
                        "Border Terrier",
                        "Kerry Blue Terrier",
                        "Irish Terrier",
                        "Norfolk Terrier",
                        "Norwich Terrier",
                        "Yorkshire Terrier",
                        "Wirehaired Fox Terrier",
                        "Lakeland Terrier",
                        "Sealyham Terrier",
                        "Airedale",
                        "Cairn",
                        "Australian Terrier",
                        "Dandi Dinmont",
                        "Boston Bull",
                        "Miniature Schnauzer",
                        "Giant Schnauzer",
                        "Standard Schnauzer",
                        "Scotch Terrier",
                        "Tibetan Terrier",
                        "Silky Terrier",
                        "Soft-coated Wheaten Terrier",
                        "West Highland White Terrier",
                        "Lhasa",
                        "Flat-coated Retriever",
                        "Curly-coater Retriever",
                        "Golden Retriever",
                        "Labrador Retriever",
                        "Chesapeake Bay Retriever",
                        "German Short-haired Pointer",
                        "Vizsla",
                        "English Setter",
                        "Irish Setter",
                        "Gordon Setter",
                        "Brittany",
                        "Clumber",
                        "English Springer Spaniel",
                        "Welsh Springer Spaniel",
                        "Cocker Spaniel",
                        "Sussex Spaniel",
                        "Irish Water Spaniel",
                        "Kuvasz",
                        "Schipperke",
                        "Groenendael",
                        "Malinois",
                        "Briard",
                        "Kelpie",
                        "Komondor",
                        "Old English Sheepdog",
                        "Shetland Sheepdog",
                        "Collie",
                        "Border Collie",
                        "Bouvier des Flandres",
                        "Rottweiler",
                        "German Shepard",
                        "Doberman",
                        "Miniature Pinscher",
                        "Greater Swiss Mountain Dog",
                        "Bernese Mountain Dog",
                        "Appenzeller",
                        "EntleBucher",
                        "Boxer",
                        "Bull Mastiff",
                        "Tibetan Mastiff",
                        "French Bulldog",
                        "Great Dane",
                        "Saint Bernard",
                        "Eskimo Dog",
                        "Malamute",
                        "Siberian Husky",
                        "Affenpinscher",
                        "Basenji",
                        "Pug",
                        "Leonberg",
                        "Newfoundland",
                        "Great Pyrenees",
                        "Samoyed",
                        "Pomeranian",
                        "Chow",
                        "Keeshond",
                        "Brabancon Griffon",
                        "Pembroke",
                        "Cardigan",
                        "Toy Poodle",
                        "Miniature Poodle",
                        "Standard Poodle",
                        "Mexican Hairless",
                        "Dingo",
                        "Dhole",
                        "African Hunting Dog"]

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, target_class = self._flat_breed_images[index]
        image_path = join(self.images_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.cropped:
            image = image.crop(self._flat_breed_annotations[index][1])

        if self.target_transform:
            target_class = self.target_transform(target_class)

        if self.transform:
            image = self.transform(image)

        if self.transform == None and self.train == True:
            img = image
            img = transforms.Resize((448, 448), Image.BILINEAR)(img)
#             img = transforms.RandomCrop((448, 448))(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            image = img

        if self.transform == None and self.train == False:
            img = image
            img = transforms.Resize((448, 448), Image.BILINEAR)(img)
#             img = transforms.CenterCrop((448, 448))(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            image = img

        return image, target_class

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, 'Images')) and os.path.exists(join(self.root, 'Annotation')):
            if len(os.listdir(join(self.root, 'Images'))) == len(os.listdir(join(self.root, 'Annotation'))) == 120:
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + join(self.root, tar_filename))
            with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(join(self.root, tar_filename))

    @staticmethod
    def get_boxes(path):
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                          int(objs.find('bndbox').find('ymin').text),
                          int(objs.find('bndbox').find('xmax').text),
                          int(objs.find('bndbox').find('ymax').text)])
        return boxes

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(join(self.root, 'lists/train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'lists/train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(join(self.root, 'lists/test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'lists/test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0] - 1 for item in labels]
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self._flat_breed_images)):
            image_name, target_class = self._flat_breed_images[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)" % (len(self._flat_breed_images), len(counts.keys()),
                                                                     float(len(self._flat_breed_images)) / float(
                                                                         len(counts.keys()))))

        return counts