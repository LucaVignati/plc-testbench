import sys

# TODO: move all hashing functions to this file

def get_class(class_name):
    '''
    This function returns the class with the given name.
    '''
    for module in sys.modules:
        if module.startswith('plctestbench'):
            if hasattr(sys.modules[module], class_name):
                return getattr(sys.modules[module], class_name)