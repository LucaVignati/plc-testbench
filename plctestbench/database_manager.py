from pathlib import Path
import pymongo
from pymongo import MongoClient
from plctestbench.node import Node
from plctestbench.utils import escape_email

class Singleton (type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
  
class DatabaseManager(metaclass=Singleton):

    def __init__(self, ip: str = None, port: int = None, username: str = None, password: str = None, user: dict = None, conn_string: str = None) -> None:
        if (ip is None or port is None or username is None or password is None or user is None) and conn_string is None:
            raise Exception("DatabaseManager: missing parameters")
        self.username = username
        self.password = password
        self.email = escape_email(user['email'])
        if conn_string:
            self.client = MongoClient(conn_string)
        else:
            self.client = MongoClient(
                host=ip,
                port=port,
                username=self.username,
                password=self.password,
            )
        self._check_if_already_initialized()
        self.save_user(user)

    def get_database(self):
        return self.client[self.email]

    def add_node(self, entry, collection_name):
        '''
        This function is used to add a node to the database.
        '''
        database = self.get_database()
        try:
            database[collection_name].insert_one(entry)
        except pymongo.errors.DuplicateKeyError as e:
            if entry['persistent']:
                raise e
            else:
                database[collection_name].replace_one({"_id": entry["_id"]}, entry)

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
        if child_collection is not None:
            for child in list(database[child_collection].find({"parent": node_id})):
                self.delete_node(child["_id"])
        filepath = Path(database[collection_name].find_one({"_id": node_id})['filepath'])
        if filepath.exists():
            filepath.unlink()
        database[collection_name].delete_one({"_id": node_id})
        database['runs'].update_many({}, {"$pull": {'nodes': {"_id": node_id}}})

    def save_run(self, run):
        '''
        This function is used to save a run to the database.
        '''
        database = self.get_database()
        try:
            database["runs"].insert_one(run)
        except pymongo.errors.DuplicateKeyError:
            print("Run already exists in the database.")

    def get_run(self, run_id):
        '''
        This function is used to retrieve a run from the database.
        '''
        database = self.get_database()
        return database["runs"].find_one({"_id": run_id})

    def set_run_status(self, run_id, status):
        '''
        This function is used to set the status of a run in the database.
        '''
        database = self.get_database()
        database["runs"].update_one({"_id": run_id}, {"$set": {"status": status}})

    def delete_run(self, run_id):
        '''
        This function is used to delete a run from the database.
        '''
        database = self.get_database()
        database["runs"].delete_one({"_id": run_id})

    def save_user(self, user):
        '''
        This function is used to save a user to the database.
        '''
        database = self.client["global"]
        if database["users"].find_one({"email": user["email"]}) is None:
            database["users"].insert_one(user)
        else:
            print("User already exists in the database.")

    def delete_user(self, email):
        '''
        This function is used to delete a user from the database.
        '''
        self.client.drop_database(escape_email(email))
        database = self.client["global"]
        database["users"].delete_one({'email': email})

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
            if self.get_database()[collection].find_one({"_id": node_id}) is not None:
                return collection
        return None

    def _check_if_already_initialized(self) -> None:
        '''
        This function is used to check if the database has already been
        initialized.
        '''
        self.initialized = False
        for collection in self.get_database().list_collection_names():
            if self.get_database()[collection].find_one({}, {"child_collection": 1}) is not None:
                self.initialized |= True
