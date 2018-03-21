#!/usr/bin/python
import os
import dense_correspondence_manipulation.utils.utils as utils

if __name__ == "__main__":
        # install ply if do not already have it
    dc_source_dir = utils.getDenseCorrespondenceSourceDir()

    ply_to_ascii_executable = os.path.join(dc_source_dir, 'src', 'ply', 'ply2ascii')
    # path_to_ply = path_to_labelfusion + "/src/ply"
    # if not (os.path.isdir(path_to_ply)):
    #     os.system("cd " + path_to_labelfusion + " && mkdir src && cd src && git clone https://github.com/peteflorence/ply.git") 
    #     os.system("cd " + path_to_ply + " && make") 

    # ply_binary_filename = lcmlog_filename + ".ply"

    ply_binary_filename = 'images.ply'
    correct_ply_header_file = os.path.join(dc_source_dir, 'config', 'correct_ply_header.txt')

    # call ply2ascii
    os.system(ply_to_ascii_executable +  "<./" + ply_binary_filename + "> ./converted_to_ascii.ply")

    # change header to be compatible with Director
    # TODO: make so Director accepts other header?
    line_elements_vertex = ""
    with open("./converted_to_ascii_modified_header.ply", 'w') as outfile:
        with open("./converted_to_ascii.ply") as infile:
            counter = 0
            for line in infile:
                counter +=1
                if counter == 3:
                    line_elements_vertex = line
                    break
        with open(correct_ply_header_file) as infile:
            counter = 0
            for line in infile:
                counter += 1
                if counter == 4:
                    outfile.write(line_elements_vertex)
                    continue
                outfile.write(line)
        with open("./converted_to_ascii.ply") as infile:
            num_skip = 14
            counter = 0
            for line in infile:
                counter += 1
                if counter <= 14:
                    continue
                outfile.write(line)

    # convert to vtp
    convert_ply_to_vtp_script = os.path.join(dc_source_dir, 'modules',
        'dense_correspondence_manipulation', 'scripts', 'convertPlyToVtp.py')

    print "converted to ascii ply format"

    os.system("directorPython " + convert_ply_to_vtp_script +  " ./converted_to_ascii_modified_header.ply")

    print "finished convert_ply_to_vtp_script"

    # clean up and rename
    # os.system("rm *.ply *.freiburg")
    os.system("mv converted_to_ascii_modified_header.vtp images.vtp")