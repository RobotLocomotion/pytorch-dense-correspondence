#!/usr/bin/python
import os
import shutil

def run_on_single_folder(folder):
    mesh_filename = os.path.join(folder, 'fusion_mesh.ply')

    if not os.path.isfile(mesh_filename):
        print "this folder is already in new format, skipping"
        return

    move_to_processed  = []
    move_to_raw = []
    for file in os.listdir(folder):

        if file in ["processed", "raw"]:
            continue
        if file == "fusion.bag":
            move_to_raw.append(file)
        else:
            move_to_processed.append(file)

    print "move_to_processed:", move_to_processed
    print "move_to_raw:", move_to_raw




    processed = os.path.join(folder, "processed")
    raw = os.path.join(folder, "raw")

    if not os.path.isdir(processed):
        os.makedirs(processed)
        os.makedirs(raw)

    for file in move_to_raw:
        src = os.path.join(folder, file)
        dest = os.path.join(raw, file)
        shutil.move(src, dest)

    for file in move_to_processed:
        src = os.path.join(folder, file)
        dest = os.path.join(processed, file)
        shutil.move(src, dest)

def main():
    logs_proto = "/home/manuelli/code/data_volume/pdc/logs_proto"
    folder = "/home/manuelli/code/data_volume/pdc/logs_proto/00_background"
    # run_on_single_folder(folder)

    for file in os.listdir(logs_proto):
        print "folder name:", file
        folder = os.path.join(logs_proto, file)
        run_on_single_folder(folder)

if __name__ == "__main__":
    main()
