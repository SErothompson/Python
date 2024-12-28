import sqlite3

class DatabaseManager:
    def __init__(self, db_name):
        self.db_name = db_name
        
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.commit()
        self.conn.close()

    def execute(self, query):
        self.cursor.execute(query)
        
    def execute_query(self, query, params=()):
        self.cursor.execute(query, params)
        return self.cursor.fetchall()
    
    def execute_many(self, query, params_list):
        self.cursor.executemany(query, params_list)
        self.conn.commit()

# Usage
with DatabaseManager('mydatabase.db') as db:
    db.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, age INTEGER)')
    db.execute('INSERT INTO users (name, age) VALUES ("John", 30)')
    result = db.execute_query('SELECT * FROM users WHERE age > ?', (25,))
    print(result)
