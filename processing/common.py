"""
This module contains commonly used functions and utilities
"""

def get_map_email(email_list: list) -> list:
    """ Maps the emails to integer ids
    An example of an email in the catalogue is '173e8bda4d6cca316ae3ea23f8d920eda60e84a9'.
    For convenience, the function will replace it with an id with an integer value
    Args:
        email_list: list of emails
    Returns:
        list of ids [int]
    """
    email_unique = list(set(email_list))
    map_dict = {email_unique[i]: i for i in range(len(email_unique))}
    return map_dict