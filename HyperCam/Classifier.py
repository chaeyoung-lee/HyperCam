"""
ML classifier implementation
"""

import os
import tqdm
import torch

from HyperCam.BSC import *
from HyperCam.Encoder import *
from utils import *

from torch.utils.cpp_extension import load

_fasthd = load(
    name="fasthd",
    extra_cflags=["-O3"],
    is_python_module=True,
    sources=[
        os.path.join("HyperCam", "onlinehd.cpp"),
    ],
)


class Classifier:

    def __init__(self, config):
        # set seed
        np.random.seed(0)
        torch.manual_seed(0)

        # configuration
        self.n = config["hv_length"]
        self.classes = config["class"]
        self.encoder = Encoder(config)
        self.item_memory = torch.zeros(self.classes, self.n)
        self.store = config.get("store", 0)
        self.config = config

        # training parameters
        if "classifier" in config:
            self.trainer = config["classifier"].get("trainer", "onlinehd")
            self.iters = config["classifier"].get("iteration", 20)
            self.lr = config["classifier"].get("learning_rate", 1)
            self.batch_size = config["classifier"].get("batch_size", 1024)
            self.bootstrap = config["classifier"].get("bootstrap", 1.0)

    def load_or_store(self, x):
        h = None
        # Load if needed
        if self.store == -1:
            import pickle

            with open(
                "{}-{}.pkl".format(self.config["encoding"], self.config["data"]), "rb"
            ) as f:
                h = pickle.load(f)
        else:
            h = [self.encoder.encode(image) for image in tqdm.tqdm(x)]

        # Store if needed
        if self.store == 1:
            import pickle

            with open(
                "{}-{}.pkl".format(self.config["encoding"], self.config["data"]), "wb"
            ) as f:
                pickle.dump(h, f)

        return h

    def train(self, x, y):
        if self.trainer == "vanilla":
            self.train_vanilla(x, y)
        elif self.trainer == "onlinehd":
            self.train_onlinehd(x, y)
        else:
            print("Unsupported trainer, running vanilla trainer.")
            self.train_vanilla(x, y)

    def train_vanilla(self, x, y):
        print("Encoding begins")
        h = self.load_or_store(x)

        print("Training... ", end="")
        dataset = {}
        for i in range(self.classes):
            dataset[i] = []
        for i in range(len(x)):
            dataset[y[i]].append(h[i])
        for label in dataset.keys():
            label_hv = BSC.bundle(dataset[label])
            self.item_memory[label] = torch.tensor(label_hv)
        print("complete!")

    def train_onlinehd(self, x, y):
        print("Encoding begins")
        h = self.load_or_store(x)

        print("Train begins")
        h = torch.tensor(h, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int8).long()
        cut = math.ceil(self.bootstrap * h.size(0))
        h_ = h[:cut]
        y_ = y[:cut]

        # updates each class hypervector (accumulating h_)
        for lbl in range(self.classes):
            self.item_memory[lbl].add_(h_[y_ == lbl].sum(0), alpha=self.lr)
        # banned will store already seen data to avoid using it later
        banned = torch.arange(cut, device=h.device)

        # todo will store not used before data
        n = h.size(0)
        todo = torch.ones(n, dtype=torch.bool, device=h.device)
        todo[banned] = False

        # will execute one pass learning with data not used during model
        # bootstrap
        h_ = h[todo]
        y_ = y[todo]
        self.item_memory = _fasthd.onepass(h_, y_, self.item_memory, self.lr)
        n = h.size(0)
        for _ in tqdm.tqdm(range(self.iters)):
            for i in range(0, n, self.batch_size):
                h_ = h[i : i + self.batch_size]
                y_ = y[i : i + self.batch_size]
                scores = self.score(h_, self.item_memory)
                y_pred = scores.argmax(1)
                wrong = y_ != y_pred

                # computes alphas to update model
                aranged = torch.arange(h_.size(0), device=h_.device)
                alpha1 = (1.0 - scores[aranged, y_]).unsqueeze_(1)
                alpha2 = (scores[aranged, y_pred] - 1.0).unsqueeze_(1)

                for lbl in y_.unique():  # for every class
                    m1 = wrong & (y_ == lbl)  # mask of missed true lbl
                    m2 = wrong & (y_pred == lbl)  # mask of wrong preds
                    self.item_memory[lbl] += self.lr * (alpha1[m1] * h_[m1]).sum(0)
                    self.item_memory[lbl] += self.lr * (alpha2[m2] * h_[m2]).sum(0)
        print("Training complete!")

    def predict(self, image):
        hv = self.encoder.encode(image)
        result = [BSC.distance(hv, class_hv) for class_hv in self.item_memory]
        label = np.argmin(result)
        return label, result[label]

    def test(self, x, y):
        print("Test begins")
        # h = self.encoder.encode(x)
        h = [self.encoder.encode(image) for image in tqdm.tqdm(x)]
        h = torch.tensor(h, dtype=torch.float32)
        scores = self.score(h, self.item_memory).argmax(1)
        accuracy = (scores == torch.tensor(y)).float().mean()
        print("ACCURACY: %f" % (accuracy))

        # Log results
        with open("results.txt", "a") as f:
            f.write("{}\t{}\n".format(self.lr, accuracy))

        return accuracy

    def score(self, x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-8):
        eps = torch.tensor(eps, device=x1.device)
        norms1 = x1.norm(dim=1).unsqueeze_(1).max(eps)
        norms2 = x2.norm(dim=1).unsqueeze_(0).max(eps)
        cdist = x1 @ x2.T
        cdist.div_(norms1).div_(norms2)
        return cdist
