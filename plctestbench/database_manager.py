from pathlib import Path
from abc import ABCMeta, abstractmethod
import pymongo
from pymongo import MongoClient
from pymongo import errors
from pymongo.errors import DuplicateKeyError
from tinydb import TinyDB, where, operations
from tempfile import NamedTemporaryFile
from datetime import datetime
from plctestbench.node import Node
from plctestbench.utils import escape_email

class Singleton (ABCMeta):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
  
class DatabaseManager(metaclass=Singleton):

    def __init__(self, ip: str | None = None,
                 port: int | None = None,
                 username: str | None = None,
                 password: str | None = None,
                 user: dict | None = None,
                 conn_string: str | None = None) -> None:
        
        if (ip is None or port is None or username is None or password is None or user is None) and conn_string is None:
            raise Exception("DatabaseManager: missing parameters")
        self.initialized = False
        self.email = escape_email(user['email']) if user is not None else None
        self._init_client(ip, port, username, password, user)

    @abstractmethod
    def _init_client(self, ip: str | None = None,
                     port: int | None = None,
                     username: str | None = None,
                     password: str | None = None,
                     user: dict | None = None,
                     conn_string: str | None = None) -> None:
        raise NotImplementedError('To be overridden!')

    @abstractmethod
    def get_database(self):
        raise NotImplementedError('To be overridden!')

    @abstractmethod
    def add_node(self, entry, collection_name):
        raise NotImplementedError('To be overridden!')

    @abstractmethod
    def find_node(self, node_id, collection_name):
        raise NotImplementedError('To be overridden!')

    @abstractmethod
    def delete_node(self, node_id):
        raise NotImplementedError('To be overridden!')

    @abstractmethod
    def save_run(self, run):
        raise NotImplementedError('To be overridden!')

    @abstractmethod
    def get_run(self, run_id):
        raise NotImplementedError('To be overridden!')

    @abstractmethod
    def set_run_status(self, run_id, status):
        raise NotImplementedError('To be overridden!')

    @abstractmethod
    def delete_run(self, run_id):
        raise NotImplementedError('To be overridden!')

    @abstractmethod
    def save_user(self, user):
        raise NotImplementedError('To be overridden!')

    @abstractmethod
    def delete_user(self, email):
        raise NotImplementedError('To be overridden!')

    @abstractmethod
    def get_child_collection(self, collection_name):
        raise NotImplementedError('To be overridden!')
 
    @abstractmethod   
    def get_collection(self, node_id):
        raise NotImplementedError('To be overridden!')

    @abstractmethod
    def _check_if_already_initialized(self) -> None:
        raise NotImplementedError('To be overridden!')
  
class MongoDatabaseManager(DatabaseManager):

    def _init_client(self, ip: str | None = None,
                     port: int | None = None,
                     username: str | None = None,
                     password: str | None = None,
                     user: dict | None = None,
                     conn_string: str | None = None) -> None:
        
        self.username = username
        self.password = password
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
        if self.email is None:
            raise Exception("DatabaseManager: email not set")
        return self.client[self.email]

    def add_node(self, entry, collection_name):
        '''
        This function is used to add a node to the database.
        '''
        database = self.get_database()
        try:
            database[collection_name].insert_one(entry)
        except DuplicateKeyError as e:
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
        if collection_name is not None:
            doc = database[collection_name].find_one({"_id": node_id})
            if doc and 'filepath' in doc:
                filepath = Path(doc['filepath'])
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
        except DuplicateKeyError:
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
        This function is used to retrieve the collection of the children of a node.
        '''
        child_collection = self.get_database()[collection_name].find_one({}, {"child_collection": 1})
        if child_collection is not None and 'child_collection' in child_collection:
            return child_collection["child_collection"]
        else:
            return None
    
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
        This function is used to check if the database has already been initialized.
        '''
        self.initialized = False
        for collection in self.get_database().list_collection_names():
            if self.get_database()[collection].find_one({}, {"child_collection": 1}) is not None:
                self.initialized |= True

class TinyDBDatabaseManager(DatabaseManager):

    def _init_client(self, ip: str | None = None,
                 port: int | None = None,
                 username: str | None = None,
                 password: str | None = None,
                 user: dict | None = None,
                 conn_string: str | None = None) -> None:
        self.client: dict[str, TinyDB] = {}

    def get_database(self, db_name: str | None = None):
        if db_name is None:
            db_name = self.email
        if db_name is None:
            raise ValueError("db_name darf nicht None sein!")
        if db_name not in self.client:
            with NamedTemporaryFile(prefix=f"{db_name}_", suffix=".json", delete=False) as tmp:
                self.client[db_name] = TinyDB(tmp.name)
        return self.client[db_name]

    def add_node(self, entry, collection_name: str):
        '''
        This function is used to add a node to the database.
        '''
        database: TinyDB = self.get_database()
        database.table(collection_name).insert(entry)

    def find_node(self, node_id, collection_name):
        '''
        This function is used to find a node in the database.
        '''
        database: TinyDB = self.get_database()
        return database.table(collection_name).get(where("_id") == node_id)

    def delete_node(self, node_id):
        '''
        This function is used to propagate the deletion of a document to its children.
        '''
        collection_name = self.get_collection(node_id)
        if isinstance(node_id, Node):
            node_id = node_id.get_id()
        database = self.get_database()
        child_collection = self.get_child_collection(collection_name)
        if child_collection is not None:
            for child in list(database.table(child_collection).search(where("parent") == node_id)):
                self.delete_node(child["_id"])

        if collection_name is not None:
            doc = database.table(collection_name).get(where("_id") == node_id)
            if doc is not None and 'filepath' in doc:
                if isinstance(doc, dict) and 'filepath' in doc:
                    filepath = Path(doc['filepath'])
                    if filepath.exists():
                        filepath.unlink()

        if collection_name is not None:
            database.table(collection_name).remove(where("_id") == node_id)
            for doc in database.table("runs").all():
                updated_nodes = [node for node in doc['nodes'] if node['_id'] != node_id]
                database.table(collection_name).update({'nodes': updated_nodes}, where("_id") == node_id)

    def save_run(self, run):
        '''
        This function is used to save a run to the database.
        '''
        database = self.get_database()
        database.table("runs").insert(self._serialize_run(run))

    def get_run(self, run_id):
        '''
        This function is used to retrieve a run from the database.
        '''
        database = self.get_database()
        return self._deserialize_run(database.table("runs").get(where("_id") == run_id))

    def set_run_status(self, run_id, status):
        '''
        This function is used to set the status of a run in the database.
        '''
        database = self.get_database()
        database.table("runs").update(operations.set("status", status), where("_id") == run_id)

    def delete_run(self, run_id):
        '''
        This function is used to delete a run from the database.
        '''
        database: TinyDB = self.get_database()
        database.table("runs").remove(where("_id") == run_id)

    def save_user(self, user):
        '''
        This function is used to save a user to the database.
        '''
        database = self.get_database("global")
        if database.table("users").search(where("email") == user["email"]) is None:
            database.table("users").insert(user)
        else:
            print("User already exists in the database.")

    def delete_user(self, email):
        '''
        This function is used to delete a user from the database.
        '''
        self.client.drop_database(escape_email(email)) # type: ignore
        database = self.get_database("global")
        database.table("users").remove(where("email") == email)

    def get_child_collection(self, collection_name):
        '''
        This function is used to retrieve the collection of the children of a
        node.
        '''
        child_collection_list = [doc for doc in self.get_database().table(collection_name).all() if "child_collection" in doc]
        if child_collection_list:
            return child_collection_list[0]["child_collection"]
        else:
            return None
    
    def get_collection(self, node_id):
        '''
        This function is used to retrieve the collection of a node.
        '''
        for collection in self.get_database().tables():
            if self.get_database().table(collection).contains(doc_id=node_id) is not None:
                return collection
        return None
        
    def _check_if_already_initialized(self) -> None:
        '''
        This function is used to check if the database has already been
        initialized.
        '''
        self.initialized = False
        for collection in self.get_database().tables():
            if [doc["child_collection"] for doc in self.get_database().table(collection).all()] != []:
                self.initialized |= True
    
    def _serialize_run(self, run):
        run["created_on"] = run["created_on"].isoformat()
        return run

    def _deserialize_run(self, run):
        run["created_on"] = datetime.fromisoformat(run["created_on"])
        return run
