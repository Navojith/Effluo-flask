from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

def get_engine():
    return create_engine('postgresql://postgres:pawara2000@localhost:5432/Effluo')

def create_session():
    return sessionmaker(bind=get_engine())

def get_session():
    return sessionmaker(bind=get_engine())()