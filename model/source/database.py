import pymysql
from pymysql.cursors import Cursor, DictCursor

class Database(object):
    def __init__(self, hostname, username, password, dbname, 
                 port, charset):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.dbname = dbname
        self.port = port
        self.charset = charset
        self.cursorclass = DictCursor
        
    def connect(self):
        self.conn = pymysql.connect(host=self.hostname,
                                    user=self.username,
                                    password=self.password,
                                    db=self.dbname,
                                    port=self.port, 
                                    charset=self.charset,
                                    cursorclass=self.cursorclass)

    def query(self, sql):
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
        except:
            print('establishing connection to database...')
            self.connect()
            print('connection established')
            cursor = self.conn.cursor()
            cursor.execute(sql)
        return cursor