import os
import zipfile
import logging

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Train:
    def __init__(self, force_download=False):
        self.force_download = force_download
        self.data_path = os.path.join(os.getcwd(), 'classifier/data/')
        self.model_path = os.path.join(os.getcwd(), 'classifier/models/')
        self.dataset_name = 'paultimothymooney/chest-xray-pneumonia'
        self.model: Sequential
        self.val: ImageDataGenerator
        self.train: ImageDataGenerator
        self.test: ImageDataGenerator
        self.epochs = 5
        
        def download_data(self):
        if self.force_download or (not os.path.isdir(self.data_path) and not os.path.isdir(self.models)):
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(self.dataset_name, path=self.data_path)
            with zipfile.ZipFile(os.path.join(self.data_path, 'chest-xray-pneumonia.zip')) as z:
                z.extractall(self.data_path)

                def load_data(self):
        self.download_data()
        train_generator = ImageDataGenerator(rescale=1/255.0)
        test_generator = ImageDataGenerator(rescale=1/255.0)
        val_generator = ImageDataGenerator(rescale=1/255.0)
        self.train = train_generator.flow_from_directory(os.path.join(self.data_path, 'chest_xray/chest_xray/train'),
                                                         target_size=(64, 64),
                                                         batch_size=32,
                                                         color_mode='grayscale',
                                                         class_mode='binary')
        self.test = test_generator.flow_from_directory(os.path.join(self.data_path, 'chest_xray/chest_xray/test'),
                                                         target_size=(64, 64),
                                                         batch_size=32,
                                                         color_mode='grayscale',
                                                         class_mode='binary')

        self.val = val_generator.flow_from_directory(os.path.join(self.data_path, 'chest_xray/chest_xray/test'),
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     color_mode='grayscale',
                                                     class_mode='binary')
        logger.info('{} images in training set; {} are PNEUMONIA'.format(self.train.samples, sum(self.train.labels)))
        
        def train_model(self):
        self.load_data()
        steps_per_epoch = self.train.samples // self.train.batch_size
        validation_steps = self.val.samples // self.val.batch_size
        self.model = self.define_model()
        self.model.fit_generator(self.train,
                                 epochs=self.epochs,
                                 validation_data=self.val,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_steps=validation_steps)
        
