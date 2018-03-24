import json
import os
import sys
import tarfile
import time
import math

import cv2
import numpy as np
import tensorflow as tf

import requests
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from stuff.helper import (FPS2, SessionWorker,  # to import in this version
                          WebcamVideoStream)
from tqdm import tqdm


class PretrainedModel:
    """ Pretrained model class. """

    def __init__(self, logger, config):
        self.logger = logger(__name__)
        self.model_name = config['source_model_name']
        self.source_url = config['source_model_url']
        self.source_tar_dir = config['source_tar_dir']
        self.source_model_dir = config['source_model_dir']
        self.source_model_file_name = config['source_model_file_name']

        self._fetch_model()

    def __repr__(self):
        return '{}({!r},{!r})'.format(
            self.__class__.__name__,
            self.logger, None)

    def __unicode__(self):
        return u'a pretrained model named {}'.format(
            self.model_name)

    def __str__(self):
        return unicode(self).encode('utf-8')

    @property
    def model_tar_file_name(self):
        return self.model_name + '.tar.gz'

    @property
    def tar_file_path(self):
        """ Get tar file path. """
        return os.path.join(self.source_tar_dir, self.model_tar_file_name)

    @property
    def model_file_path(self):
        """ Get model file path. """
        return os.path.join(self.source_model_dir, self.source_model_file_name)

    @property
    def model_url(self):
        """ Get complete model url. """
        return self.source_url + self.model_tar_file_name

    def _download_file(self):
        """ Download pretrained model from the internet. """
        self.logger.info('Downloading model now')
        request = requests.get(self.model_url, stream=True)

        total_size = int(request.headers.get('content-length', 0))
        block_size = 1024
        wrote = 0

        with open(self.tar_file_path, 'wb') as downloaded_file:
            pbar = tqdm(request.iter_content(block_size), total=math.ceil(
                total_size//block_size), unit='KB', unit_scale=True)

            for data in pbar:
                wrote += len(data)
                downloaded_file.write(data)
        if total_size != 0 and wrote != total_size:
            self.logger.error("ERROR, something went wrong")

    def _extract_tar(self):
        """ Extract downloaded model file. """
        tar_file = tarfile.open(self.tar_file_path)

        for file_ in tar_file:
            file_name = os.path.basename(file_.name)
            if self.source_model_file_name in file_name:
                tar_file.extract(self.source_model_dir)
        os.remove(tar_file)

    def _fetch_model(self):
        """ Fetch model from model zoo. """
        if not os.path.isfile(self.model_file_path):
            self.logger.info('Model files not found')
            self._download_file()
            self._extract_tar()
        else:
            self.logger.info('Model files found, Proceed')


class ConfigParser:
    """ A JSON file config parser custom module. """

    def __init__(self, logger, config_file_path):
        self.logger = logger(__name__)
        self.config_file_path = config_file_path
        self._parse_file()

    def __repr__(self):
        return '{}({!r},{!r})'.format(
            self.__class__.__name__,
            self.logger, self.config_file_path)

    def __unicode__(self):
        return u'a json parser for file {}'.format(
            self.config_file_path)

    def __str__(self):
        return unicode(self).encode('utf-8')

    def _parse_file(self):
        """ Parse file for JSON data. """
        self.logger.info('Parsing file for config')
        with open(self.config_file_path) as json_file:
            self._parsed_data = json.load(json_file)

    def parse_model_config(self):
        try:
            return self._parsed_data['model_config']
        except:
            raise KeyError('model_config must be defined in config file')

    def parse_detection_config(self):
        try:
            return self._parsed_data['detect_config']
        except:
            raise KeyError('detect_config must be defined in config file')


class Logger:
    # TODO:Implement or use tensorflow built in ???
    pass


class ObjectDetection:

    def __init__(self, logger, config_file_path):
        self.logger = logger(__name__)
        self._load_config(logger, config_file_path)
        self._load_model(logger, self._detect_config)

    def _load_config(self, logger, config_file_path):
        config_parser = ConfigParser(logger, config_file_path)
        self._model_config = config_parser.parse_model_config()
        self._detect_config = config_parser.parse_detection_config()

    def _load_model(self, logger, model_config):
        model = PretrainedModel(self.logger, model_config)
