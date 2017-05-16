#!/usr/bin/python
from configparser import ConfigParser
 
 
def config(filename='database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    parser[u'postgresql'] = {'host':'cribhubdb.c7bhn1c0aibh.us-east-2.rds.amazonaws.com',
                      'database':'cribhubdb',
                      'user':'cribhub',
                      'password':'cs4300project'}
    # read config file actually not working so don't do it
    #parser.read(filename)
    #print(parser.sections())
    #print(parser.has_section(section))
    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
 
    return db
