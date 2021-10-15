import os
from os import path
import glob as gb
from ecctestbench.node import Node

folder_suffixes = ['lost_samples_masks',
                   'ecc_tracks',
                   'output_analyses']

def compute_absolute_folder_path(node: Node) -> str:
    '''
    This private function computes the absolute path of the given
    node climbing down the ancestors list.

        Inputs:
            node:   the node instance to compute the absolute path for
    '''
    folder_path = ""
    for ancestor in node.ancestors:
        folder_path = path.join(folder_path, ancestor.get_folder_name())
    return folder_path

class PathManager(object):

    def __init__(self, root_folder) -> None:
        '''
        This class defines and creates the folder structure used to store
        all the data of the program. It also provides utility functions
        to deal with filepaths.

            Inputs:
                root_folder:    The root folder where all the files will
                                be created
        '''
        self.root_folder = root_folder
        if not path.exists(root_folder):
            print("The folder %s does not exist", root_folder)
            return

        self.original_tracks_paths = gb.glob(self.root_folder + '/*.wav')

    def get_original_tracks(self) -> list:
        return self.original_tracks_paths

    def set_root_node_path(node: Node) -> None:
        '''
        This function computes the absolute path of the root node, creates
        its directory and stores it in the node.

            Inputs:
                node:   the node instance to compute the absolute path for
        '''
        track_path = node.get_file().name
        folder_name = track_path.split('.wav')[0] + '-' + folder_suffixes[0]
        node.set_folder_name(folder_name)
        if not path.exists(folder_name):
            os.mkdir(folder_name)

    def set_node_relative_path(node: Node) -> None:
        '''
        This function computes the relative path of the node, creates its
        directory and stores it in the node.

        Inputs:
            node:   the node instance to compute the relative path for
        '''
        index = node.depth
        if index < len(folder_suffixes):
            node_path = compute_absolute_folder_path(node)
            worker_name = str(node.get_worker())
            folder_name = worker_name + '-' + folder_suffixes[index]
            folder_path = path.join(node_path, folder_name)
            node.set_folder_name(folder_name)
            if not path.exists(folder_path):
                os.mkdir(folder_path)

    def get_file_path(node: Node) -> str:
        '''
        This function computes the absolute path of the given node and
        returns it.

            Inputs:
                node:   the node instance to compute the absolute path for
        '''
        abs_folder_path = compute_absolute_folder_path(node)
        filename = str(node.get_worker())
        return path.join(abs_folder_path, filename)

    def change_file_extension(filepath: str, new_extension: str) -> str:
        '''
        This function retrieves the filepath associated to the given node and
        returns the same filepath changing its extension to the given one.

            Inputs:
                filepath:       the filepath to apply the new extension to
                new_extension:  the extension to apply to the new filepath
        '''
        if '.' not in new_extension:
            new_extension = '.' + new_extension
        filepath_no_extension, _ = filepath.split('.')
        return filepath_no_extension + new_extension
