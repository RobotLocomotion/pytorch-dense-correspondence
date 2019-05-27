import os
import sys
import yaml

print sys.argv

if len(sys.argv) < 2:
    print "expected arg to specify which composite dataset .yaml to download data for"
    print "for example:"
    print "    python download_pdc_data.py /path/to/pytorch-dense-correspondence/config/dense_correspondence/dataset/composite/caterpillar_only.yaml"
    quit()


logs = []
datasets_to_download_config = yaml.load(file(sys.argv[1]))
pdc_root_dir = os.getcwd()


if len(sys.argv) >= 3:
    data_dir = sys.argv[2]
else:
    data_dir = pdc_root_dir

pdc_data_dir = os.path.join(data_dir, "pdc")
print "data_dir", data_dir
print "pdc_data_dir", pdc_data_dir

def add_datset(path):
    global logs
    single_object_dataset = yaml.load(file(path))
    for j in single_object_dataset["train"]:
        if j not in logs:
            logs.append(j)
    for j in single_object_dataset["test"]:
        if j not in logs:
            logs.append(j)

for i in datasets_to_download_config['single_object_scenes_config_files']:
    path = os.path.join(pdc_root_dir, "config/dense_correspondence/dataset/single_object", i)
    add_datset(path)

for i in datasets_to_download_config['multi_object_scenes_config_files']:
    path = os.path.join(pdc_root_dir, "config/dense_correspondence/dataset/multi_object", i)
    add_datset(path)



if not os.path.isdir(pdc_data_dir):
    os.makedirs(pdc_data_dir)


download_url_base = "https://data.csail.mit.edu/labelfusion/pdccompressed/"


if not os.path.isdir(os.path.join(pdc_data_dir, "evaluation_labeled_data")):
    print "downloading labeled data"
    os.chdir(data_dir)
    # note the evaluation_labeled_data_compressed.tar.gz" actually has the form
    # pdc/evaluation_labeled_data so you need to be careful when unpacking it
    os.system("wget -q "+download_url_base+"evaluation_labeled_data_compressed.tar.gz")
    evaluation_data_compressed_path = os.path.join(data_dir, "evaluation_labeled_data_compressed.tar.gz")
    os.system("tar -zxf " + evaluation_data_compressed_path)
    os.system("rm " + evaluation_data_compressed_path)
else:
    print "already have labeled data -- skipping ..."


print "Will download these logs if do not already have:"
print logs

logs_proto_dir = os.path.join(pdc_data_dir, "logs_proto")
if not os.path.isdir(logs_proto_dir):
    os.makedirs(logs_proto_dir)

## uncompress each log
for i, log in enumerate(logs):
    os.chdir(pdc_data_dir)
    log_path = log+".tar.gz"
    log_path = os.path.join(logs_proto_dir, log)
    log_compressed_path = os.path.join(logs_proto_dir, log+".tar.gz")
    if os.path.isdir(os.path.join(logs_proto_dir, log)):
        print "already have log",log," -- skipping ..."
        continue
    print "downloading   log", i+1, "of", len(logs)
    print "log_compressed_path", log_compressed_path
    os.system("rm -rf " + log_compressed_path)
    
    # change to logs_proto dir
    os.chdir(logs_proto_dir)

    # download the compressed log
    os.system("wget " + download_url_base+"logs_proto/"+log + ".tar.gz")

    # extract the log
    os.chdir(data_dir)
    os.system("tar -zvxf" + log_compressed_path)
    os.system("rm -rf " + log_compressed_path)

print "done"
