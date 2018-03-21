#!/usr/bin/python
import os
import dense_correspondence_manipulation.utils.utils as utils


def run(data_folder, ply_binary_filename='images.ply'):

    # install ply if do not already have it
    vtp_filename = os.path.join(data_folder, 'images.vtp')
    dc_source_dir = utils.getDenseCorrespondenceSourceDir()

    ply_to_ascii_executable = os.path.join(dc_source_dir, 'src', 'ply', 'ply2ascii')
    path_to_ply = os.path.join(dc_source_dir, "src", "ply")
    if not (os.path.isfile(ply_to_ascii_executable)):
        os.system("cd " + path_to_ply + " && make")


    correct_ply_header_file = os.path.join(dc_source_dir, 'config', 'correct_ply_header.txt')

    ply_binary_full_filename = os.path.join(data_folder, ply_binary_filename)
    converted_ascii_filename = os.path.join(data_folder, "converted_to_ascii.ply")
    converted_ascii_modified_header_filename = os.path.join(data_folder, "converted_to_ascii_modified_header.ply")

    # call ply2ascii
    os.system(ply_to_ascii_executable + "<./" + ply_binary_filename + "> " + converted_ascii_filename)

    # change header to be compatible with Director
    # TODO: make so Director accepts other header?
    line_elements_vertex = ""
    with open(converted_ascii_modified_header_filename, 'w') as outfile:
        with open(converted_ascii_filename) as infile:
            counter = 0
            for line in infile:
                counter += 1
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
        with open(converted_ascii_filename) as infile:
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

    os.system("directorPython " + convert_ply_to_vtp_script + " " + converted_ascii_modified_header_filename)


    converted_ascii_modified_header_vtp_filename = os.path.join(data_folder, "converted_to_ascii_modified_header.vtp")

    print "finished convert_ply_to_vtp_script"

    # clean up and rename
    # os.system("rm *.ply *.freiburg")
    os.rename(converted_ascii_modified_header_vtp_filename, vtp_filename)



if __name__ == "__main__":
    run(os.getcwd())