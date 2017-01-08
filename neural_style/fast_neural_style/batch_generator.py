import sys
from multiprocessing import Process, Queue
from random import sample

import numpy as np
from keras.preprocessing.image import list_pictures

from neural_style.utils import load_and_preprocess_img

DEFAULT_MAX_QSIZE = 1000


class BatchGenerator:

    def __init__(self, imdir, num_batches, batch_size, image_size, max_qsize=None):
        max_qsize = max_qsize if max_qsize is not None else DEFAULT_MAX_QSIZE
        self.batchq = Queue(max_qsize)
        self.generator_process = Process(target=BatchGenerator.generate_batches, args=(self.batchq, imdir, num_batches, batch_size, image_size))
        self.generator_process.start()
        self.consumed_batches = 0
        self.num_batches = num_batches

    def get_batch(self):
        if self.consumed_batches == self.num_batches:
            raise StopIteration
        else:
            self.consumed_batches += 1
            return self.batchq.get()

    @staticmethod
    def generate_batches(batchq, imdir, num_batches, batch_size, image_size):
        image_paths = list_pictures(imdir)
        if not image_paths:
            print("Error: no images found in {}".format(imdir))
            sys.exit(1)
        for _  in range(num_batches):
            batch_image_paths = sample(image_paths, batch_size)
            batch = np.vstack([load_and_preprocess_img(image_path, image_size, center_crop=True) for image_path in batch_image_paths])
            batchq.put(batch)

