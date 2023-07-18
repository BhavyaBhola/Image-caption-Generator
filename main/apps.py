from django.apps import AppConfig
from keras.models import load_model
import pickle
import os
from django.conf import settings
from keras.applications.vgg16 import VGG16 , preprocess_input
from keras.models import Model

class MainConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'

class TokenizerConfig(AppConfig):
    name='tokenizer'
    token = os.path.join(settings.MODEL, "tokenizer.pkl")

    with open(token , 'rb') as f:
        tokenzizer = pickle.load(f)

class CaptionModelConfig(AppConfig):
    name='caption_gen'
    model_path = os.path.join(settings.MODEL, "caption_gen_v1.h5")
    model = load_model(model_path)

class FeatureExtModelConfig(AppConfig):
    name='feature_ext'
    pre_trained_model = VGG16()
    feature_extractor = Model(inputs = pre_trained_model.inputs , outputs = pre_trained_model.layers[-2].output)