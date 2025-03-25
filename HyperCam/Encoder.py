"""
HyperCam encoder implementation
"""

import math
from copy import deepcopy
from HyperCam.BSC import *


class Encoder:

    def __init__(self, config):
        # set seed
        np.random.seed(0)

        # class variables from configuration
        self.w = config["width"]
        self.h = config["height"]
        self.p = config["pixel"]
        self.n = config["hv_length"]
        self.p_r = config["pixel_resolution"]
        self.encoding = config["encoding"]

        # codebook
        self.codebook = None
        self.pixelbook = self.generate_level_vectors(int(self.p / self.p_r))
        if self.encoding == "naive":
            self.codebook = np.random.randint(2, size=(self.w * self.h, self.n))
        elif self.encoding == "hypercam":
            self.codebook = np.random.randint(
                2, size=(int(math.ceil(self.w * self.h / self.n)), self.n)
            )
        elif self.encoding == "hypercam-countsketch":
            self.d = config["density"]
            codebook = []
            for _ in range(int(math.ceil(self.w * self.h / self.n))):
                ps = sorted(np.random.choice(self.n, self.d, replace=False))
                codebook.append((ps, np.random.choice([-1, 1], self.d)))
            self.codebook = np.array(codebook)
            self.deterministic = np.random.randint(2, size=self.n)
        elif self.encoding == "hypercam-bloomfilter":
            self.d = config["density"]
            codebook = []
            for _ in range(int(math.ceil(self.w * self.h / self.n))):
                ps = sorted(np.random.choice(self.n, self.d, replace=False))
                codebook.append(ps)
            self.codebook = codebook
            self.deterministic = np.random.randint(2, size=self.n)

    def generate_level_vectors(self, size):
        hvs = []
        now = np.zeros(self.n, dtype=np.int8)
        block = int(self.n / size)
        xor_pos = list(range(self.n))
        xor_pos = np.random.permutation(xor_pos)
        for i in range(size):
            hvs.append(deepcopy(now))
            for pos in xor_pos[i * block : min(i * block + block, self.n)]:
                now[pos] ^= 1
        return np.array(hvs)

    def encode(self, vars):
        if self.encoding == "naive":
            return self.encode_naive(vars)
        elif self.encoding == "hypercam":
            return self.encode_hypercam(vars)
        elif self.encoding == "hypercam-countsketch":
            return self.encode_count_sketch(vars)
        elif self.encoding == "hypercam-bloomfilter":
            return self.encode_bloom_filter(vars)

    def encode_naive(self, image):
        hv = np.bitwise_xor(self.codebook, self.pixelbook[image])
        hv = np.mean(hv, axis=0)
        result = np.zeros(self.n, dtype=np.int8)
        result[hv > 0.5] = 1
        result[hv < 0.5] = 0
        equal_indices = hv == 0.5
        result[equal_indices] = np.random.randint(2, size=np.sum(equal_indices))
        return result

    def encode_hypercam(self, image):
        img_hv = []
        k, cb_id = 0, 0
        while k < self.w * self.h:
            pos_hv = self.codebook[cb_id]
            kk = 0
            while kk < self.n and k + kk < self.w * self.h:
                v = int(image[k + kk] / self.p_r)
                img_hv.append(BSC.bind([pos_hv, self.pixelbook[v]]))
                pos_hv = BSC.permute(pos_hv, 1)
                kk += 1
            k += kk
            cb_id += 1
        return BSC.bundle(img_hv)

    def encode_count_sketch(self, image):
        img_hv = np.zeros((self.p, self.n), dtype=np.int16)
        s = np.zeros((self.p, self.n), dtype=np.int16)
        ws = np.zeros(self.p, dtype=np.int16)
        i, cb_id = 0, 0
        while i < self.w * self.h:
            ii = 0
            while ii < self.n and i + ii < self.w * self.h:
                v = int(image[i + ii] / self.p_r)
                for j in range(self.d):
                    s[v][(ii + self.codebook[cb_id][0][j]) % self.n] += self.codebook[
                        cb_id
                    ][1][j]
                ws[v] += 1
                ii += 1
            i += ii
            cb_id += 1
        for v in range(self.p):
            s[v] = np.where(s[v] == 0, self.deterministic, np.where(s[v] > 0, 1, 0))
            hv = BSC.bind([s[v], self.pixelbook[v]])
            img_hv[v] = hv
        return BSC.bundle_with_weights(img_hv, ws)

    def encode_bloom_filter(self, image):
        img_hv = []
        s = [np.zeros(self.n, dtype=np.int16) for _ in range(self.p)]
        ws = np.zeros(self.p, dtype=np.int16)
        i, cb_id = 0, 0
        while i < self.w * self.h:
            ii = 0
            while ii < self.n and i + ii < self.w * self.h:
                v = int(image[i + ii] / self.p_r)
                for j in range(self.d):
                    s[v][(ii + self.codebook[cb_id][j]) % self.n] = 1
                ws[v] += 1
                ii += 1
            i += ii
            cb_id += 1
        for v in range(self.p):
            hv = BSC.bind([s[v], self.pixelbook[v]])
            img_hv.append(hv)
        return BSC.bundle_with_weights(img_hv, ws)
