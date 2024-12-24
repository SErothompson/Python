import mysql.connector
from mysql.connector import Error

class MySQLDatabase:
    def __init__(self, host_name, user_name, user_password, db_name):
        self.connection = self.create_connection(host_name, user_name, user_password, db_name)

    def create_connection(self, host_name, user_name, user_password, db_name):
        connection = None
        try:
            connection = mysql.connector.connect(
                host=host_name,
                user=user_name,
                passwd=user_password,
                database=db_name
            )
            print("Connection to MySQL DB successful")
        except Error as e:
            print(f"The error '{e}' occurred")
        
        return connection

    def execute_query(self, query):
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            self.connection.commit()
            print("Query executed successfully")
        except Error as e:
            print(f"The error '{e}' occurred")
        finally:
            cursor.close()

    def execute_read_query(self, query):
        cursor = self.connection.cursor()
        result = None
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            return result
        except Error as e:
            print(f"The error '{e}' occurred")
        finally:
            cursor.close()

        return result

    def execute_insert_query(self, query):
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            self.connection.commit()
            print("Query executed successfully")
        except Error as e:
            print(f"The error '{e}' occurred")
        finally:
            cursor.close()

# Example usage
db = MySQLDatabase("localhost", "root", "zAQ!08091983", "test")