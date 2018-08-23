import os

import sys

import yaml



print sys.argv



if len(sys.argv) < 2:

    print "expected arg to specify which subset of data to download"

    print "for example:"

    print "    python download_pdc_data.py /something"

    quit()



logs = []

datasets_to_download_config = yaml.load(file(sys.argv[1]))

for i in datasets_to_download_config['single_object_scenes_config_files']:

    i_path = os.path.dirname(os.path.dirname(sys.argv[1]))+"/single_object/"+i

    single_object_dataset = yaml.load(file(i_path))

    print single_object_dataset

    for j in single_object_dataset["train"]:

        if j not in logs:

            logs.append(j)

    for j in single_object_dataset["test"]:

        if j not in logs:

            logs.append(j)

        

os.system("mkdir -p ./pdc")



download_url_base = "https://data.csail.mit.edu/labelfusion/pdccompressed/"



if not os.path.isdir("./pdc/evaluation_labeled_data"):

    print "downloading labeled data"

    os.system("cd pdc && wget -q "+download_url_base+"evaluation_labeled_data_compressed.tar.gz")

    evaluation_data_compressed_path = "./pdc/evaluation_labeled_data_compressed.tar.gz"

    os.system("tar -zxf " + evaluation_data_compressed_path)

    os.system("rm " + evaluation_data_compressed_path)

else:

    print "already have labeled data -- skipping ..."





print "Will download these logs if do not already have:"

print logs

os.system("mkdir -p ./pdc/logs_proto")



## uncompress each log

for i, log in enumerate(logs):

    log_path = log+".tar.gz"

    if os.path.isdir("pdc/logs_proto/"+log):

        print "already have log",log," -- skipping ..."

        continue

    print "downloading   log", i+1, "of", len(logs)

    os.system('rm -f pdc/logs_proto/'+log_path)

    os.system("cd pdc/logs_proto && wget "+download_url_base+"logs_proto/"+log_path+" && cd ../../ && tar -zvxf pdc/logs_proto/"+log_path+" && rm pdc/logs_proto/"+log_path)

    #print "uncompressing log", i+1, "of", len(logs)

    #os.system("tar -zxf " + os.path.join("pdc",log_path))

    #os.system("rm "+os.path.join("pdc",log_path))



print "done"
