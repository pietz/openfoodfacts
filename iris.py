"""IRIS - Intelligent Reverse Image Search

Author: Paul-Louis Pröve
Affiliation: Lufthansa Industry Solutions
Version: 0.13.1
"""

import base64
import io
import os
import time
from glob import glob

import feather
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as tfs
from PIL import Image
from sklearn.decomposition import PCA
from torchvision.models import resnet18
from tqdm import tqdm


class IRIS:
    """Intelligent Reverse Image Search

    The IRIS class contains a set of functionalites to build a
    reverse image search system with a high level API

    Attributes:
        meta: A pandas DataFrame containing additional meta
            information about the images. One column mus be named
            `path` and will ne identical to the `paths` attribute.
            meta will be not None if IRIS was instantiated with a
            pandas DataFrame.
        paths: A list of images paths
        embeds: A 2D embedding array. First dim represents the
            number of images and second dim represents the embedding
            size.
        model: An optional PyTorch model that takes a batch of
            transformed images (output of `tf`) and outputs an
            embedding vector for each.
        tf: A PyTorch transforms function that takes a single RGB
            PIL image and outputs a fitting input to the `model`.
            When using the default model, this should output a
            W x H x C PyTorch tensor.
        rotations: A list of rotation degrees that should be performed
            for the generation of each embedding. The resulting vector
            will be the sum of every rotated image embedding. This
            is helpful when images may be initially wrongly rotated.
        pca: A boolean indicating whether or not pca should be used,
            an integer indicating the n_components of the pca that is
            to be performed, a path to a joblib file containing a
            scikit-learn PCA object or a scikit-learn PCA object.
        device: PyTorch related attribute that manages the use of the
            GPU when available and the CPU otherwise.
    """

    def __init__(
        self, inputs, model=None, tf=None, rotations=[0], pca=None, square_mode="crop"
    ):
        """Inits IRIS with attributes

        inputs: A path to a directory containing a set of images,
            a path to a .feather file containing a snapshot of IRIS,
            a list of images paths or a pandas DataFrame containing
            a column named `path`
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model)
        self.tf = self._load_tf(tf)
        self.rotations = rotations
        self.pca = None
        self.square_mode = square_mode
        self.paths, self.embeds, self.meta = self._load_inputs(inputs)
        self.pca = self._load_pca(pca)

    def __len__(self):
        return len(self.paths)

    def _load_inputs(self, inputs, rmv_dups=False):
        """Sets the paths, meta and embeds attribute

        Args:
            inputs: A path to a directory containing a set of images,
                a path to a .feather file containing a snapshot of IRIS,
                a list of images paths or a pandas DataFrame containing
                a column named `path`.
            rmv_dups: Aka `remove duplicates`. When set to True will
                only handle entries that don't currently exist in the
                IRIS instance.

        Returns:
            meta: A pandas DataFrame containing additional meta
                information about the images. One column mus be named
                `path` and will ne identical to the `paths` attribute.
                meta will be not None if IRIS was instantiated with a
                pandas DataFrame.
            paths: A list of images paths
            embeds: A 2D embedding array. First dim represents the
                number of images and second dim represents the embedding
                size.
        """
        embeds, meta = None, None
        if isinstance(inputs, str) and inputs.endswith(".feather"):
            # inputs a path to a .feather file
            paths, embeds, meta = self._load_feather(inputs)
            if rmv_dups:
                idx = [x not in self.paths for x in paths]
                paths = paths[idx]
                embeds = embeds[idx]
                if meta is not None:
                    meta = meta.iloc[idx]
        elif isinstance(inputs, str):
            # inputs is a directory
            paths = self._glob_paths(inputs)
            if rmv_dups:
                idx = [x not in self.paths for x in paths]
                paths = paths[idx]
            embeds = self._load_embeds(paths)
        elif isinstance(inputs, pd.DataFrame):
            # inputs is a pandas DataFrame
            meta = inputs
            assert "path" in meta.columns
            paths = meta["path"].values
            if rmv_dups:
                idx = [x not in self.paths for x in paths]
                paths = paths[idx]
            embeds = self._load_embeds(paths)
        elif isinstance(inputs, (list, np.ndarray)):
            # inputs is a list/array of image paths
            paths = np.array(inputs)
            if rmv_dups:
                idx = [x not in self.paths for x in paths]
                paths = paths[idx]
            embeds = self._load_embeds(paths)
        else:
            raise ValueError

        return paths, embeds, meta

    def _glob_paths(self, directory):
        """Searches for images in a given path

        Args:
            directory: Path to look for images

        Retuns:
            Numpy array of images paths
        """
        li = []
        for ext in ["jpg", "JPG", "jpeg", "JPEG"]:
            li += glob(os.path.join(directory, "**/*." + ext), recursive=True)
        return np.array(sorted(li))

    def _load_feather(self, path):
        """Loads a feather file and translates it to class attributes

        Args:
            path: Path to the feather file

        Returns:
            meta: A pandas DataFrame containing additional meta
                information about the images. One column mus be named
                `path` and will ne identical to the `paths` attribute.
                meta will be not None if IRIS was instantiated with a
                pandas DataFrame.
            paths: A list of images paths
            embeds: A 2D embedding array. First dim represents the
                number of images and second dim represents the embedding
                size.
        """
        df = pd.read_feather(path)
        embed_dim = int(df.columns[-1]) + 1
        embeds = df.iloc[:, -1 * embed_dim :].to_numpy().astype(np.float32)
        paths = df["path"].values
        if len(df.columns) > embed_dim + 1:
            meta = df.iloc[:, : -1 * embed_dim]
        else:
            meta = None
        return paths, embeds, meta

    def _load_model(self, model):
        """Loads the neural network model for predictions

        Args:
            model: Either a path to a persisted PyTorch model or
                when None will load a ResNet18 that was pretrained
                on the ImageNet dataset. The latter requires either
                an internet connection or that the ResNet weights
                are available locally on disk.
        Returns:
            A PyTorch model set in eval mode
        """
        if isinstance(model, str):
            # model is a path to a PyTorch model
            model = torch.load(self.model)
        elif model is None:
            # use default model
            model = resnet18(pretrained=True)
            model.fc = nn.Identity()
        model.to(self.device)
        model.eval()
        return model

    def _load_tf(self, tf, size=224):
        """Loads the image transformation needed for the model

        A PyTorch model only accepts PyTorch tensors as inputs. As
        such we need to convert a native PIL image object to a
        PyTorch tensor. Additional resizing, cropping and
        normalization is done to match the expected input size.

        Args:
            tf: A PyTorch transforms function
            size: A custom image size. Should only be changed if the
                model is also changed.

        Returns: A PyTorch transforms function
        """
        if tf is None:
            tf = tfs.Compose(
                [
                    tfs.Resize(size),
                    tfs.CenterCrop(size),
                    tfs.ToTensor(),
                    tfs.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        return tf

    def _load_embeds(self, paths):
        """Calculates embeddings for the collection of images

        Once the collection of images is set, this function will
        create the necessary embeddings for the search functionality.
        This function might run for multiple hours if the collection
        is larger than 100k images. The process can be sped up
        dramatically (10-50x) when using a GPU with Cuda.

        Args:
            paths: A list of image paths

        Returns:
            A 2D array of embeddings
        """
        print("Calculating Embeddings...")
        embeds = []
        for path in tqdm(paths):
            img = self.load_img(path)
            embed = self.get_embedding(img)
            embeds.append(embed)
        embeds = np.array(embeds)
        return embeds

    def _load_pca(self, pca, dim=128):
        """Instantiates or loads PCA if necessary

        Args:
            pca: A boolean indicating whether or not pca should be used,
                an integer indicating the n_components of the pca that is
                to be performed, a path to a joblib file containing a
                scikit-learn PCA object or a scikit-learn PCA object.
            dim: int that will be used for n_components if `pca` is True

        Returns: A scitkit-learn PCA instance or None
        """
        if pca is None or pca is False:
            # pca should not be used
            return None
        if isinstance(pca, (int, bool)):
            # pca is True or an int specifying n_components
            dim = pca if isinstance(pca, int) else dim
            pca = PCA(n_components=dim, whiten=True)
            pca.fit(self.embeds)
        elif isinstance(pca, str):
            pca = joblib.load(pca)
        if pca.n_components_ != self.embeds.shape[-1]:
            self.embeds = pca.transform(self.embeds)
        return pca

    def load_img(self, img, square_img=True):
        """Loads an image in different possible formats

        Args:
            img: An integer indicating the ith position in the
                current list of images, an image path, a Base64
                image string or a PIL image.
            crop_square: Whether or not to crop the image to a 1:1
                aspect ratio. Often needed for neural networks

        Returns: A PIL image
        """
        if isinstance(img, (int, np.integer)):
            img = self.paths[img]
        if isinstance(img, str):
            if img.startswith("data:image/jp"):
                img = self.b64img2pil(img)
            else:
                try:
                    img = Image.open(img).convert("RGB")
                except:
                    # Return Dummy Image
                    img = Image.fromarray(np.zeros((400, 400, 3), dtype=np.uint8))

        if square_img:
            if self.square_mode == "crop":
                img = self.center_crop(img)
            else:
                size = min(img.size()[0], img.size()[1])
                img = img.resize((size, size), Image.BILINEAR)
        return img

    def save(self, path=""):
        """Saves the current state of IRIS to disk

        Merges the embedding array and the meta data to a single
        pandas DataFrame and saves it to disk as a feather file.

        Args:
            path: optional path where to save the file. If empty
                string will save to root folder. Name of the file
                will be iris_<TIMECODE>.feather
        """
        if len(path) == 0 or os.path.isdir(path):
            # use standard name
            name = "iris_" + time.strftime("%y%m%d%H%M%S")
            name += ".feather"
            path = os.path.join(path, name)
        df_e = pd.DataFrame(self.embeds)
        if self.meta is None:
            df_m = pd.DataFrame({"path": self.paths})
        else:
            df_m = self.meta
        feather.write_dataframe(df_m.join(df_e), path)

    def get_embedding(self, img):
        """Calculates the embedding for an image

        Args:
            img: param acceptable to load_img()

        Returns:
            A Numpy embedding vector
        """
        img = self.load_img(img)
        imgs = [self.tf(img.rotate(x)) for x in self.rotations]
        imgs = torch.stack(imgs)
        with torch.no_grad():
            pred = self.model(imgs.to(self.device)).mean(dim=0)
        pred = pred.detach().cpu().numpy()
        if isinstance(self.pca, PCA):
            pred = self.pca.transform(pred[None, ...])[0]
        return pred

    def center_crop(self, img):
        """Crops a PIL image to 1:1 aspect ratio

        Args:
            img: A PIL image

        Returns:
            A square cropped PIL image
        """
        width, height = img.size
        size = width if width < height else height
        left, top = (width - size) // 2, (height - size) // 2
        return img.crop((left, top, left + size, top + size))

    def shorter_resize(self, img, size=224):
        """Resizes an image given the shorter side length

        Args:
            img: A PIL image
            size: Pixel length of the shorter side

        Returns:
            A PIL image
        """
        w, h = img.size
        if w > h:
            ratio = w / h
            return img.resize((int(size * ratio), size), Image.BILINEAR)
        else:
            ratio = h / w
            return img.resize((size, int(size * ratio)), Image.BILINEAR)

    def cosine_similarity(self, query, fil):
        """
        Calculates the cosine similarity between a query image and
        all images in the collection.

        Args:
            query: The query embedding vector
            fil: Filter to perform a subselection on the collection

        Returns:
            The cosine similarity between the query and each image
                in the image collection.
        """
        q_dot = np.sqrt(query @ query)
        e_dot = np.sqrt((self.embeds[fil] ** 2).sum(axis=1))
        return np.dot(query, self.embeds[fil].T) / (q_dot * e_dot)

    def combine_inputs(self, query, add_inp):
        """Combines multiple mebeddings into one by taking the mean

        Args:
            query: The query embedding vector
            add_inp: Additional inputs specified as a list of collection
                IDs or as a 2D array of embeddings.
        Returns:
            The mean of all the inputs
        """
        if isinstance(add_inp, (list, int, np.integer)):
            add_inp = self.embeds[add_inp]
        query = np.array(query)
        if len(query.shape) == 1:
            query = query[None, ...]
        add_inp = np.array(add_inp)
        if len(add_inp.shape) == 1:
            add_inp = add_inp[None, ...]
        query = np.concatenate([query, add_inp], axis=0)
        return query.mean(axis=0)

    def b64img2pil(self, b64):
        """Converts a Base64 image to PIL image"""
        b64_body = b64.split(",")[1]
        decoded = base64.b64decode(b64_body)
        raw = io.BytesIO(decoded)
        return Image.open(raw).convert("RGB")

    def pilimg2b64(self, img, jpg_quality=80):
        """Converts a PIL image to Base64 image"""
        buff = io.BytesIO()
        img.save(buff, format="JPEG", quality=jpg_quality)
        b64_body = base64.b64encode(buff.getvalue()).decode()
        b64_head = "data:image/jpg;base64,"
        return b64_head + b64_body

    def fil2idx(self, fil):
        """Converts a filter to a list of collection IDs

        If additional meta information are provided in the form of
        a pandas DataFrame, `fil` can be a dict mapping column names
        to a list of acceptable values. This function converts this
        dict to a list of IDs that match the criteria.

        Args:
            fil: Can either be a list of IDs or a dict

        Returns:
            A list of collection IDs
        """
        if isinstance(fil, (list, np.ndarray)):
            # is already a list of indices
            return np.array(fil, dtype=int)
        elif isinstance(fil, dict):
            # dict maps column name to filtered values
            li = []
            for col, val in fil.items():
                if not isinstance(val, (list, np.ndarray)):
                    val = [val]
                li.append(self.meta[col].isin(val).values)
            li = np.array(li)
            li = li.all(axis=0)
            return np.array([i for i, val in enumerate(li) if val])
        else:
            raise ValueError

    def show(self, img):
        """Helpful for Jupyter notebooks to quickly show an image"""
        return self.load_img(img, square_img=False)

    def update(self, inputs):
        """Updates the current state of IRIS with new images

        For this we don't want to calculate existing embeddings again
        which is why we check for duplicates first.

        Args:
            inputs: A path to a directory containing a set of images,
                a path to a .feather file containing a snapshot of IRIS,
                a list of images paths or a pandas DataFrame containing
                a column named `path`
        """
        p, e, m = self._load_inputs(inputs, rmv_dups=True)
        if len(p) > 0:
            self.paths = np.concatenate([self.paths, p])
            self.embeds = np.concatenate([self.embeds, e])
            if self.meta is not None and m is not None:
                self.meta = pd.concat([self.meta, m], ignore_index=True)

    def search(self, inp, add_inp=None, fil=None, k=10, rtype="index", size=150):
        """Searches the collection for similar images

        This is the core function of IRIS. The input is converted to
        an embedding, is being merged with additional inputs, is filtered
        if necessary and cross checked with the collection.

        Args:
            inp: The input file. See load_img() for details
            add_inp: Additonal inputs. See combine_inputs()
                for details
            fil: A filter to limit the search. See fil2idx()
                for details
            k: Number of results to return
            rtype: A single return type or list of return types.
                Possible options:
                - index: ID in the collection
                - distance: distance to the query
                - path: path to the images
                - embed: embedding of the results
                - image: PIL image of the results
                - base64: Base64 string of the images
            size: Size of the returned images in case rtype includes
                image or base64
        Returns:
            Either a list of results in case of a single rtype or
            a dict of multiple lists, where the key is the rtype
            name.
        """
        if fil is None:
            fil = np.arange(len(self.paths))
        if isinstance(inp, np.ndarray):
            # img is already an embedding
            q = inp
        else:
            img = self.load_img(inp)

            q = self.get_embedding(img)

        if add_inp is not None and len(add_inp) > 0:
            q = self.combine_inputs(q, add_inp)

        fil = self.fil2idx(fil)
        dist = 1 - self.cosine_similarity(q, fil)
        idx = dist.argsort()
        dist = dist[idx]
        idx = fil[idx]

        di = {}
        di["index"] = list(idx[:k])
        di["distance"] = list(dist[:k])
        di["embed"] = list(self.embeds[idx[:k]])
        if self.meta is None:
            di["path"] = [self.paths[i] for i in idx[:k]]
        else:
            for col in self.meta:
                di[col] = self.meta.loc[idx[:k], col].values
        if "image" in rtype or "base64" in rtype:
            di["image"] = [self.load_img(x) for x in di["path"]]
            di["image"] = [self.shorter_resize(x, size) for x in di["image"]]
        if "base64" in rtype:
            di["base64"] = [self.pilimg2b64(x) for x in di["image"]]

        if isinstance(rtype, str):
            return di[rtype]
        # return [di[x] for x in rtype]
        return {k: di[k] for k in rtype}
