
#!/usr/bin/python
import psycopg2
import json
import urllib2
from config import config
from Loader import listing_id_to_listing_db
from CribHub import CribHub 
class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = np.ascontiguousarray(obj)
                assert(cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            data_b64 = base64.b64encode(obj_data)
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
                # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)

def json_numpy_obj_hook(dct):
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct
 
def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()
 
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
 
        # create a cursor
        cur = conn.cursor()
        
        # execute a statement
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')
 
        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)
       
        return conn, cur
     # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')
            
create_table_request = """"""
 
def create_tables():
    """ create tables in the PostgreSQL database"""
    commands = (
        """
        CREATE TABLE listingid_to_text_english(
            listing_id serial PRIMARY KEY,
            reviews VARCHAR NOT NULL );
        """,)
    conn = None
    try:
        # read the connection parameters
        params = config()
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        # create table one by one
        for command in commands:
            cur.execute(command)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            
def insert_values(listingid_to_text):
    """ insert multiple vendors into the vendors table  """
    sql = "INSERT INTO listingid_to_text_english VALUES (%s, %s)"
    args = [(key, val) for key, val in listingid_to_text.iteritems()]
    conn = None
    try:
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        
        print("here")
        # execute the INSERT statement
        cur.executemany(sql, args)
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
 
def get_text(listing_id):
    """ query parts from the parts table """
    conn = None
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        cur.execute("SELECT reviews FROM listingid_to_text WHERE listing_id IN (%s)", [listing_id])
        rows = cur.fetchall()
        print(rows)
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

def create_reviews_table():
    """ create tables in the PostgreSQL database"""
    commands = (
        """
        CREATE TABLE airbnb_review_to_vec(
            review_id VARCHAR PRIMARY KEY,
            vec VARCHAR NOT NULL );
        """,)
    conn = None
    try:
        # read the connection parameters
        params = config()
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        # create table one by one
        for command in commands:
            cur.execute(command)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

def insert_airbnb_values_to_review_table(airbnb_review_id_to_vec_dict):
    """ insert multiple vendors into the vendors table  """
    sql = "INSERT INTO airbnb_review_to_vec VALUES (%s, %s)"
    args = [(key, val) for key, val in airbnb_review_id_to_vec_dict.iteritems()]

    conn = None
    try:
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        dataText = ','.join(cur.mongrify('(%s, %s)', row) for row in args)

        print("here")
        # execute the INSERT statement
        cur.execute("INSERT INTO airbnb_review_id_to_vec VALUES " + dataText)
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

def get_airbnb_review(listing_id):

    conn = None
    regex = str(listing_id) + "X.+"
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        cur.execute("SELECT vec FROM airbnb_review_id_to_vec WHERE (review_id REGEXP (%s))", regex)
        rows = cur.fetchall()
        print(rows)
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()


def create_reviews_table_nytimes():
    """ create tables in the PostgreSQL database"""
    commands = (
        """
        CREATE TABLE nytimes_review_to_vec(
            review_id serial PRIMARY KEY,
            vec VARCHAR NOT NULL );
        """,)
    conn = None
    try:
        # read the connection parameters
        params = config()
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        # create table one by one
        for command in commands:
            cur.execute(command)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

def insert_nyt_values_to_review_table(nyt_review_id_to_vec_dict):
    """ insert multiple vendors into the vendors table  """
    sql = "INSERT INTO nytimes_review_to_vec VALUES (%s, %s)"

    args = [(key, val) for key, val in nyt_review_id_to_vec_dict.iteritems()]
    conn = None
    try:
        # read database configuration
        params = config()
        # connect to the PostgreSQL database
        conn = psycopg2.connect(**params)
        # create a new cursor
        cur = conn.cursor()
        
        print("here")
        # execute the INSERT statement
        cur.executemany(sql, args)
        # commit the changes to the database
        conn.commit()
        # close communication with the database
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

def get_nytimes_review(review_ids):

    conn = None
    placeholders = ", ".join(str(rid) for rid in review_ids)
    print(placeholders)
    query = "SELECT * FROM nytimes_review_to_vec WHERE review_id IN (%s)" % placeholders
    try:
        params = config()
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        print(rows)
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

if __name__ == '__main__':    
    #listing_id to listing dict
    #file = urllib2.urlopen('https://s3.amazonaws.com/cribble0108/airbnb_listing_id_to_listing.json')
    #listingid_to_text = listing_id_to_listing_db()
    #create_tables()
    #insert_values(listingid_to_text)
    #get_text(2515)
    #create_reviews_table()
    #create_reviews_table_nytimes()
    #cribhub = CribHub()
    #print("here")
    #get_nytimes_review([1001, 1002])