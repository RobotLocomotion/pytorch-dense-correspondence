# system
import os

# pdc
import dense_correspondence_manipulation.utils.utils as pdc_utils

def poser_don_executable_filename():
    return os.path.join(os.getenv("POSER_BUILD_DIR"), 'apps', 'poser_don', 'poser_don_app')

def poser_source_dir():
    return os.path.join(pdc_utils.getDenseCorrespondenceSourceDir(), 'src', 'poser')

def poser_don_example_data_dir():
    return os.path.join(poser_source_dir(), 'apps', 'poser_don', 'data')

