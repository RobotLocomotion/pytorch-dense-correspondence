# Basic I/O utils
import yaml
import numpy as np
import os

def getDictFromYamlFilename(filename):
    """
    Read data from a YAML files
    """
    stream = file(filename)
    return yaml.load(stream)

def saveToYaml(data, filename):
    """
    Save a data to a YAML file
    """
    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def getDenseCorrespondenceSourceDir():
    return os.getenv("DC_SOURCE_DIR")


def dictFromPosQuat(pos, quat):
    """
    Make a dictionary from position and quaternion vectors
    """
    d = dict()
    d['translation'] = dict()
    d['translation']['x'] = pos[0]
    d['translation']['y'] = pos[1]
    d['translation']['z'] = pos[2]

    d['quaternion'] = dict()
    d['quaternion']['w'] = quat[0]
    d['quaternion']['x'] = quat[1]
    d['quaternion']['y'] = quat[2]
    d['quaternion']['z'] = quat[3]

    return d


def getQuaternionFromDict(d):
    """
    Get the quaternion from a dict describing a transform. The dict entry could be
    one of orientation, rotation, quaternion depending on the convention
    """
    quat = None
    quatNames = ['orientation', 'rotation', 'quaternion']
    for name in quatNames:
        if name in d:
            quat = d[name]


    if quat is None:
        raise ValueError("Error when trying to extract quaternion from dict, your dict doesn't contain a key in ['orientation', 'rotation', 'quaternion']")

    return quat

def getPaddedString(idx, width=6):
    return str(idx).zfill(width)