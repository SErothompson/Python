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
db = MySQLDatabase("localhost", "root", "", "test")
db.execute_query("CREATE TABLE IF NOT EXISTS users (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), age INT)")
db.execute_query("INSERT INTO users (name, age) VALUES ('John', 30)")
db.execute_query("INSERT INTO users (name, age) VALUES ('Jane', 25)")
db.execute_query("INSERT INTO users (name, age) VALUES ('Bob', 40)")
db.execute_query("INSERT INTO users (name, age) VALUES ('Alice', 35)")
db.execute_query("INSERT INTO users (name, age) VALUES ('Charlie', 45)")
db.execute_query("INSERT INTO users (name, age) VALUES ('David', 50)")
db.execute_query("INSERT INTO users (name, age) VALUES ('Eve', 55)")
db.execute_query("INSERT INTO users (name, age) VALUES ('Frank', 60)")
db.execute_query("INSERT INTO users (name, age) VALUES ('Grace', 65)")
print(db.execute_read_query("SELECT * FROM users WHERE age > 50"))