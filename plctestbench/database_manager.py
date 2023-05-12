from pymongo import MongoClient
from threading import Thread

def deletion_trigger(database):
    '''
    This function is triggered whenever a document is deleted from the database.
    It looks for all the documents that have the deleted document as parent and
    deletes them as well.
    '''

    def propagate_deletion(document):
        '''
        This function is used to propagate the deletion of a document to its
        children.
        '''
        for child in database.find({"parent": document["_id"]}):
            propagate_deletion(child)
            database.delete_one({"_id": child["_id"]})
    
    cursor = database.watch()
    while(True):
        document = next(cursor)
        if document["operationType"] == "delete":
            propagate_deletion(document["documentKey"]["_id"])
        
    

class DatabaseManager(object):

    def __init__(self, ip: str='localhost', port: str='27017') -> None:
        CONNECTION_STRING = "mongodb://" + ip + ":" + port
        self.client = MongoClient(CONNECTION_STRING)

        # Start trigger thread !!! Not working !!!
        # self.trigger_thread = Thread(target=deletion_trigger, args=(self.get_database(),))
        # self.trigger_thread.start()

    def get_database(self):
        return self.client["plc_database"]