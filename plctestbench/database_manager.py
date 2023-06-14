from pymongo import MongoClient
from plctestbench.node import Node
from pathlib import Path
  
class DatabaseManager(object):

    def __init__(self, ip: str='localhost', port: str='27017') -> None:
        CONNECTION_STRING = "mongodb://" + ip + ":" + port
        self.client = MongoClient(CONNECTION_STRING)
        self.initialized = self.check_if_already_initialized()

    def get_database(self):
        return self.client["plc_database"]
    
    def add_node(self, entry, collection_name):
        '''
        This function is used to add a node to the database.
        '''
        database = self.get_database()
        database[collection_name].insert_one(entry)

    def find_node(self, node_id, collection_name):
        '''
        This function is used to find a node in the database.
        '''
        database = self.get_database()
        return database[collection_name].find_one({"_id": node_id})
    
    def delete_node(self, node_id):
        '''
        This function is used to propagate the deletion of a document to its
        children.
        '''
        collection_name = self.get_collection(node_id)
        if isinstance(node_id, Node):
            node_id = node_id.get_id()
        database = self.get_database()
        child_collection = self.get_child_collection(collection_name)
        if child_collection!=None:
            for child in list(database[child_collection].find({"parent": node_id})):
                self.delete_node(child["_id"])
        Path(database[collection_name].find_one({"_id": node_id})['filename']).unlink()
        database[collection_name].delete_one({"_id": node_id})

    def get_child_collection(self, collection_name):
        '''
        This function is used to retrieve the collection of the children of a
        node.
        '''
        child_collection = self.get_database()[collection_name].find_one({}, {"child_collection": 1})
        return child_collection["child_collection"] if 'child_collection' in child_collection.keys() else None
    
    def get_collection(self, node_id):
        '''
        This function is used to retrieve the collection of a node.
        '''
        for collection in self.get_database().list_collection_names():
            if self.get_database()[collection].find_one({"_id": node_id}) != None:
                return collection
        return None

    def check_if_already_initialized(self):
        '''
        This function is used to check if the database has already been
        initialized.
        '''
        initialized = False
        for collection in self.get_database().list_collection_names():
            if self.get_database()[collection].find_one({}, {"child_collection": 1}) != None:
                initialized |= True
        return initialized