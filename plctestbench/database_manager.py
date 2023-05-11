from pymongo import MongoClient

class DatabaseManager(object):

    def __init__(self, db_ip: str) -> None:
        CONNECTION_STRING = "mongodb://" + db_ip + ":27017"
        self.client = MongoClient(CONNECTION_STRING)

    def get_database(self):
        return self.client["plc_database"]


# dbname = get_database()
# print(dbname)
# collection_name = dbname["user_1_items"]
# item_1 = {
#   "_id" : "U1IT00001",
#   "item_name" : "Blender",
#   "max_discount" : "10%",
#   "batch_number" : "RR450020FRG",
#   "price" : 340,
#   "category" : "kitchen appliance"
# }

# item_2 = {
#   "_id" : "U1IT00002",
#   "item_name" : "Egg",
#   "category" : "food",
#   "quantity" : 12,
#   "price" : 36,
#   "item_description" : "brown country eggs"
# }
# collection_name.insert_many([item_1,item_2])