# Basic I/O utils

def getDictFromYamlFilename(filename):
    stream = file(filename)
    return yaml.load(stream)

def saveToYaml(data, filename):
    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)