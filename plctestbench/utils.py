import sys
import hashlib

def get_class(class_name):
    '''
    This function returns the class with the given name.
    '''
    for module_name, module in sys.modules.items():
        if module_name.startswith('plctestbench'):
            if hasattr(module, class_name):
                return getattr(module, class_name)
    raise ValueError(f"The class {class_name} does not exist.")

def compute_hash(obj):
    '''
    This function returns the hash of the given object.
    '''
    return int.from_bytes(hashlib.md5(str(obj).encode('utf-8')).digest()[:7], 'little')

def escape_email(email):
    '''
    This function escapes the given email address.
    '''
    return email.replace('@', '_at_').replace('.', '_dot_')
