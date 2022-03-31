# some trainging parameters

EPOCHS = 50
BATCH_SIZE = 32
NUM_CLASSES = 6

image_height = 150
image_width = 150
channels = 3
save_model_dir = 'Tensorflow2.0_ResNet/saved_model/model'
dataset_dir = 'Tensorflow2.0_ResNet/dataset/'
train_dir = dataset_dir + 'train'
valid_dir = dataset_dir + 'valid'
test_dir = dataset_dir + 'test'

# choose a network
# model = 'resnet18'
# model = 'resnet34'
model = 'resnet50'
# model = 'resnet101'
# model = 'resnet152'