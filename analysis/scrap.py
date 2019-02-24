class Compare(object):
    @staticmethod
    def cosine_similarity(x, model):
        probs = model(x)  # (batch_size, num_classes)

        pass

    @staticmethod
    def pca_similarity(x, model, k=2):
        from sklearn.decomposition import PCA
        pca = PCA(k)
        probs = model(x)
        probs = probs.data.numpy()  # (batch_size, num_classes)
        probs_new = pca.fit_transform(probs)

        data_dict = {}

        return probs_new

    pass


#
# from adversarial_sampling_experiments.data_providers import *
#
# x, y = ImageDataGetter.mnist(
#     filename=os.path.join(globals.ROOT_DIR, 'data/mnist-test.npz'),
#     shuffle=True
# )
#
# print("x shape: ",x.shape)
# print("y shape: ",y.shape)

import numpy as np
x = np.arange(10)
print("Original array:")
print(x)
np.random.shuffle(x)
n = 1
print (x[np.argsort(x)[-2]])