import os
from os import path
import glob as gb
from plctestbench.node import Node

folder_suffixes = ['lost_samples_masks',
                   'reconstructed_tracks',
                   'output_analyses']

def compute_absolute_folder_path(parent: Node) -> str:
    '''
    This private function computes the absolute path of the given
    node climbing down the ancestors list.

        Inputs:
            node:   the node instance to compute the absolute path for
    '''
    folder_path = ""
    for ancestor in parent.ancestors:
        folder_path = path.join(folder_path, ancestor.get_folder_name())
    folder_path = path.join(folder_path, parent.get_folder_name())
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

        self.original_tracks_paths = sorted(gb.glob(self.root_folder + '/*.wav'))

    def get_original_tracks(self) -> list:
        return self.original_tracks_paths

    def _set_root_node_path(self, settings) -> tuple:
        '''
        This function computes the absolute path of the root node, creates
        its directory and stores it in the node.

            Inputs:
                node:   the node instance to compute the absolute path for
        '''
        track_path = self.root_folder + '/' + settings.get('filename')
        folder_name = track_path.split('.wav')[0] + '-' + folder_suffixes[0]
        absolute_path = track_path.split('.wav')[0]
        if not path.exists(folder_name):
            os.mkdir(folder_name)
        return folder_name, absolute_path

    def get_node_paths(self, worker, settings, parent: Node) -> tuple:
        '''
        This function computes the relative path of the node, creates its
        directory and stores it in the node.

        Inputs:
            node:   the node instance to compute the relative path for
        '''
        if parent is None:
            return self._set_root_node_path(settings)
        
        folder_name = None
        index = parent.depth + 1
        node_path = compute_absolute_folder_path(parent)
        worker_name = worker.__name__
        absolute_path = path.join(node_path, worker_name)
        if index < len(folder_suffixes):
            folder_name = worker_name + '-' + folder_suffixes[index]
            folder_path = path.join(node_path, folder_name)
            if not path.exists(folder_path):
                os.mkdir(folder_path)
        return folder_name, absolute_path

    @staticmethod
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
