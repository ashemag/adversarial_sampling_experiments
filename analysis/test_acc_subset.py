from adversarial_sampling_experiments.globals import ROOT_DIR
from adversarial_sampling_experiments.data_providers import ImageDataGetter
from adversarial_sampling_experiments.models.simple_fnn import FeedForwardNetwork
from adversarial_sampling_experiments.data_subsetter import DataSubsetter
from adversarial_sampling_experiments.model_queries import ModelQuery
import os

model = FeedForwardNetwork(img_shape=(1,28,28),num_classes=10)
model.load_model(model_path=os.path.join(ROOT_DIR,'saved_models/simple_fnn/model_epoch_49'))

x, y = ImageDataGetter.mnist(filename=os.path.join(ROOT_DIR,'data/mnist-train.npz'))
x, y = DataSubsetter.condition_on_label(x, y, labels=[2])
acc_subset = ModelQuery.accuracy(x,y,model)

print("accuracy: ", acc_subset)

