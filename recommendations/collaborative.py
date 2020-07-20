"""
This module contains the final recommendations methods
"""
from recommendations_with_IBM.read_data import reader


def recommend_for_new_user(titles=False, n_max=10):
    """ Returns recommendations for new users
    For new user there is no information to find similar users, the only recommendation possible is based on the most
    popular articles in the catalogue.
    Args:
        titles: if True will return the titles of the articles, else the article ids
        n_max: maximum number of recommendations to get
    Returns:
    Examples:
        >>> recommend_for_new_user(n_max=3)
        [1330, 1429, 1364]
    """
    return reader.UserList().get_most_popular_articles(titles=titles)[: n_max]


def recommend(user_id: int, titles=False, n_max=10):
    """ Returns a list of recommendations based on the status of the user
    Args:
        user_id:
        titles:
        n_max:
    Returns:
    Examples:
        >>> recommend(2, n_max=2)
        [1024, 1351]
        >>> recommend('a', n_max=2)
        [1330, 1429]
    """
    user_info = reader.UserList()
    user_matrix = user_info.user_matrix
    if user_id not in user_matrix.index:
        # new user
        print("User not in the list of users")
        return recommend_for_new_user(titles, n_max)
    elif user_info.get_user_status_from_id(user_id) == 'new_user':
        print("New users: not enough information for collaborative recommendation")
        return recommend_for_new_user(titles, n_max)
    else:
        return user_info.get_user_recomendations(user_id, titles, n_max)
