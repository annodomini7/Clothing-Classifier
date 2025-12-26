import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class ClothingDataset(Dataset):
    def __init__(self, dataframe, images_dir, label_encoder, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.images_dir = images_dir
        self.label_encoder = label_encoder
        self.transform = transform

        self.images = []
        self.labels = []

        for _, row in self.dataframe.iterrows():
            img_path = os.path.join(images_dir, f"{row['id']}.jpg")
            if os.path.exists(img_path):
                self.images.append(img_path)
                self.labels.append(self.label_encoder[row["articleType"]])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.images)))

        if self.transform:
            image = self.transform(image)

        return image, label


class ClothingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=32,
        num_workers=4,
        image_size=224,
        transform_mean=None,
        transform_std=None,
        val_split=0.1,
        test_split=0.1,
        min_samples_per_class=10,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_split = val_split
        self.test_split = test_split
        self.min_samples_per_class = min_samples_per_class

        self.transform_mean = transform_mean
        self.transform_std = transform_std

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
                transforms.RandomRotation(degrees=10),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.transform_mean, std=self.transform_std),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.transform_mean, std=self.transform_std),
            ]
        )

        self.label_encoder = None
        self.label_decoder = None
        self.num_classes = None
        self.class_to_category = None
        self.master_category_encoder = None

    def _load_and_filter_data(self):
        csv_path = os.path.join(self.data_dir, "styles.csv")
        images_dir = os.path.join(self.data_dir, "images")

        df = pd.read_csv(csv_path, on_bad_lines="skip")
        df = df.dropna(subset=["articleType"])

        class_counts = df["articleType"].value_counts()
        min_samples = self.min_samples_per_class
        valid_classes = class_counts[class_counts >= min_samples].index
        df = df[df["articleType"].isin(valid_classes)]
        df = df[
            df["id"].apply(
                lambda x: os.path.exists(os.path.join(images_dir, f"{x}.jpg"))
            )
        ]

        return df, images_dir

    def _create_label_encoder(self, df):
        categories = sorted(df["articleType"].unique())
        self.label_encoder = {cat: idx for idx, cat in enumerate(categories)}
        self.label_decoder = {idx: cat for cat, idx in self.label_encoder.items()}
        self.num_classes = len(categories)

        master_categories = sorted(df["masterCategory"].unique())
        self.master_category_encoder = {
            cat: idx for idx, cat in enumerate(master_categories)
        }

        sub_to_master = (
            df.drop_duplicates("articleType")
            .set_index("articleType")["masterCategory"]
            .to_dict()
        )
        self.class_to_category = {}
        for sub in categories:
            class_idx = self.label_encoder[sub]
            cat_idx = self.master_category_encoder[sub_to_master[sub]]
            self.class_to_category[class_idx] = cat_idx

    def get_num_classes(self):
        return self.num_classes

    def get_label_encoder(self):
        return self.label_encoder

    def get_label_decoder(self):
        return self.label_decoder

    def get_class_to_category(self):
        return self.class_to_category

    def get_class_to_name(self):
        return {idx: name for name, idx in self.label_encoder.items()}

    def setup(self, stage=None):
        df, images_dir = self._load_and_filter_data()
        self._create_label_encoder(df)

        train_df, temp_df = train_test_split(
            df,
            test_size=self.val_split + self.test_split,
            stratify=df["articleType"],
            random_state=42,
        )

        val_ratio = self.val_split / (self.val_split + self.test_split)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_ratio,
            stratify=temp_df["articleType"],
            random_state=42,
        )

        self.train_dataset = ClothingDataset(
            train_df,
            images_dir,
            self.label_encoder,
            transform=self.train_transform,
        )
        self.val_dataset = ClothingDataset(
            val_df,
            images_dir,
            self.label_encoder,
            transform=self.val_transform,
        )
        self.test_dataset = ClothingDataset(
            test_df,
            images_dir,
            self.label_encoder,
            transform=self.val_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
