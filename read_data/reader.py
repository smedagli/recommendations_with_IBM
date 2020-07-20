import numpy as np
import pandas as pd
pd.options.display.max_columns = 25
pd.options.display.width = 2500

from recommendations_with_IBM.processing import common as cm
from recommendations_with_IBM.info import paths



class UserList():
    def __init__(self, reduced=False):
        """ Set reduced = True to get a subsample of the dataset to test functions """
        self.file = paths.users_path
        self.df = self._load_data(reduced=reduced)
        self.article_dict = self.df.drop_duplicates()[['article_id',
                                                       'title']].set_index('article_id').to_dict()['title']
        self.user_matrix = self.get_user_matrix()
        self.similarity_matrix = self._get_user_similarities()


    def _load_data(self, reduced=False):
        """ Loads the .csv file and makes small adjustments
        Adjustments:
            * drop NaN for email
            * maps email to integers
        """
        df = pd.read_csv(self.file, index_col=0)
        # drop NaN
        df = df.dropna(subset=['email'], axis=0)
        # map email to user id
        mp = cm.get_map_email(df.email.tolist())
        df['user_id'] = df['email'].apply(lambda x: mp[x])
        if reduced:
            df = df.iloc[: 500]
        return df.astype({'article_id': int})


    def get_top_articles(self, ascending=False, titles=False, unique_readers=True) -> list:
        """ Returns the article ids as a list ordered from the most visited, to the least (default)
        Args:
            ascending: default is False, so results will be from the most to the least visited.
            titles: if True, returns the titles of the articles, otherwise will return the id
            unique_readers: if True, than multiple interaction from one user id will be accounted as only 1
        Returns:
        Examples:
            >>> UserList().get_top_articles(unique_readers=False)[: 5]
            [1429, 1330, 1431, 1427, 1364]
            >>> UserList().get_top_articles(unique_readers=True)[: 5]
            [1330, 1429, 1364, 1314, 1398]
        """
        if unique_readers:
            if titles:
                return list(map(self.get_title_from_article_id, self.get_top_articles(unique_readers=True)))
            else:
                return self.user_matrix.sum(axis=0).sort_values(ascending=False).index.tolist()
        else:
            if titles:
                return list(map(self.get_title_from_article_id, self.get_top_articles(unique_readers=False)))
            else:
                return self.df.groupby('article_id').count()['title'].sort_values(ascending=ascending).index.tolist()


    def get_title_from_article_id(self, article_id: int):
        """ Returns the title of the article given the id. If the id not in the list, will return the id again
        Examples:
            >>> UserList().get_title_from_article_id(2)
            'this week in data science (april 18, 2017)'
        """
        if article_id not in self.article_dict.keys():
            print("Article could not be found")
            return article_id
        else:
            return self.article_dict[article_id]


    def get_user_matrix(self) -> pd.DataFrame:
        """ Returns the users vs articles matrix
        Returns a matrix of "one hot encoded" articles.
        User ids [int] are the ROWS
        Article ids [int] are the COLUMNS

        From this matrix we can get also information about
            * how many articles a user read:
                >>> u = UserList()
                >>> u.user_matrix.sum(axis=1)
            * how many times an article has been read by a unique user:
                >>> u = UserList()
                >>> u.user_matrix.sum(axis=0)
        """
        user_item = pd.get_dummies(self.df, columns=['article_id']).groupby(['user_id'], as_index=True).sum()
        user_item = user_item.apply(lambda x: x != 0)
        user_item = user_item.astype(int)
        return user_item.rename(columns=lambda x: int(x.split('article_id_')[-1]))


    def get_articles_read_by_user_id(self, user_id: int) -> list:
        """ Returns the ids of the articles read by the specified user_id as a list
        Args:
            user_id:
        Returns:
            list of articles id interacted with
        Examples:
            >>> UserList().get_articles_read_by_user_id(1)
            [369, 508, 1028]
        """
        if user_id not in self.user_matrix.index.tolist():
            print(f'User {user_id} not included in the user matrix')
            return []
        else:
            subset = self.user_matrix.loc[user_id]
            return subset[subset == 1].index.tolist()


    def get_nr_of_articles_by_user_id(self, user_id: int) -> int:
        """ Returns the number of articles read by the specified user_id
        Args:
            user_id:
        Returns:
        Examples:
            >>> UserList().get_nr_of_articles_by_user_id(1)
            3
        """
        return len(self.get_articles_read_by_user_id(user_id))


    def get_unique_readers_by_article_id(self, article_id: int) -> list:
        """ Returns the ids of the users having read the specified article_id as a list
        Args:
            article_id:
        Returns:
            list of user id that have accesse the article
        Examples:
            >>> UserList().get_unique_readers_by_article_id(1444)
            [465, 1234, 3514, 3559, 4054]
        """
        if article_id not in self.user_matrix.columns.tolist():
            print(f'Article {article_id} not included in the user matrix')
            return []
        else:
            return user_matrix[user_matrix[article_id] == 1].index.tolist()


    def get_nr_of_unique_readers_by_article_id(self, article_id: int) -> int:
        """ Returns the number of unique readers of the specified article_id
        Args:
            user_id:
        Returns:
            number of unique users that have read the article
        Examples:
            >>> UserList().get_nr_of_unique_readers_by_article_id(2)
            44
        """
        return len(self.get_unique_readers_by_article_id(article_id))


    def _get_user_similarities(self) -> pd.DataFrame:
        """ Returns the correlation between the users as a dataframe
        Returns:
        """
        return pd.DataFrame(np.corrcoef(self.user_matrix),
                            index=self.user_matrix.index,
                            columns=self.user_matrix.index)


    def get_most_similar_users(self, user_id: int) -> list:
        """ Returns the user ids most similar to the specified user id as a list
        Users with a larger number of interactions have "priority" meaning that if two users have same similarity to
        user_id, the first in the ranking will be the one with most interactions.
        Args:
            user_id:
        Returns:
        Examples:
            >>> UserList().get_most_similar_users(1)[: 5]
            [4578, 4599, 1295, 1825, 1104]
        """
        return self._get_similarity_per_user(user_id).sort_values(by=['similarity', 'nr_of_interactions'],
                                                  ascending=False).index.tolist()


    def _get_similarity_per_user(self, user_id: int) -> pd.DataFrame:
        """ Returns the matrix of similarities with other users + the number of interactions from each user
        Args:
            user_id:
        Returns:
            DataFrame with ROWS the other users and COLUMNS the nr of articles read by the user ('nr_of_interactions')
            and the similarity with the user specified as input (user_id) ('similarity')
        Examples:
            >>> UserList()._get_similarity_per_user(1).nr_of_interactions.tolist()[: 2]
            [1, 33]
        """
        temp = pd.concat((self.df.groupby('user_id').article_id.count(), self.similarity_matrix.loc[user_id]),
                  axis=1).rename(columns={'article_id': 'nr_of_interactions', user_id: 'similarity'})
        return temp.drop(user_id)


    def get_user_recomendations(self, user_id: int, titles=False, n_max=10) -> list:
        """ Returns the list of suggestions for the given user, based on the similarities with other users
        Will sort the other users by similarity and add the articles read by the other users to a list of suggestions.
        Will also remove the articles already read by user_id
        Args:
            user_id:
            titles: if True will return the titles of the articles suggested, else will return the articles' ids
            n_max: maximum number of suggesions
        Returns:
        Examples:
            >>> UserList().get_user_recomendations(1, n_max=3)
            [15, 390, 1186]
            >>> UserList().get_user_recomendations(1, n_max=1, titles=True)
            ['connect to db2 warehouse on cloud and db2 using scala']
        """
        already_read = self.get_articles_read_by_user_id(1)
        if titles:
            return list(map(self.get_title_from_article_id, self.get_user_recomendations(user_id, n_max=n_max)))
        else:
            similar_users = self.get_most_similar_users(user_id)
            suggestions = []
            go_on = True
            while go_on:
                for user in similar_users:
                    suggestions += list(
                        set(self.get_articles_read_by_user_id(user)) - set(suggestions) - set(already_read))
                    if len(suggestions) >= n_max:
                        return sorted(suggestions[: n_max])
                go_on = False
            return suggestions


    def get_most_popular_articles(self, titles=False) -> list:
        """ Returns the articles sorted by the number of unique users' interactions
        Args:
            titles: if True, returns the titles of the articles, else their article id
        Returns:
        Examples:
            >>> UserList().get_most_popular_articles()[: 5]
            [1330, 1429, 1364, 1314, 1398]
            >>> UserList().get_most_popular_articles(titles=True)[0]
            'insights from new york car accident reports'
        """
        if titles:
            return list(map(self.get_title_from_article_id, self.get_most_popular_articles()))
        else:
            return self.user_matrix.sum(axis=0).sort_values(ascending=False).index.tolist()


    def get_users_with_most_article_read(self) -> list:
        """ Returns the users with most articles read (multiple readings accounted as 1)
        Returns:
        Examples:
            >>> UserList().get_users_with_most_article_read()[: 3]
            [5113, 1256, 4039]
        """
        return self.user_matrix.sum(axis=1).sort_values(ascending=False).index.tolist()


    def get_most_active_users(self) -> list:
        """ Returns the users with most articles read (multiple readings accounted as multiple values)
        Returns:
        Examples:
            >>> UserList().get_most_active_users()[ :3]
            [1256, 5113, 2454]
        """
        return self.df.groupby('user_id').count()['title'].sort_values(ascending=False).index.tolist()


    def _get_users_map_threshold(self) -> list:
        """ Returns a list of numbers thr_list used to classify users based on the number of activities
        if number of activities < thr_list[0] --> new user
        elif number of activities < thr_list[1] --> common user
        else number of activities < thr_list[2] --> superuser

        thr_list should be based on percentiles like:
            thr_list = [25 percentile, 75 percentile]
        but in this particular case is necessary to define fixed numbers.
        That's because because min/max are distant but 25-75% are really close.
        A fixed list of values will be used
        Examples:
            >>> _get_users_map_threshold()
            [9, 35]
        """
        d = self.df.groupby('user_id').title.count().describe()
        # thr_list = [d.loc['25%'], d.loc['75%']]
        # fixed list
        q95 = self.df.groupby('user_id').title.count().quantile(.95)
        thr_list = [d.loc['75%'], q95]
        return [int(x) for x in thr_list]


    def _get_status_from_number_of_articles(self, n: int) -> str:
        """ Classifies the users based on the number of articles they read n
        Examples:
            >>> UserList()._get_status_from_number_of_articles(18)
            'regular_user'
        """
        thr_list = self._get_users_map_threshold()
        if n <= thr_list[0]:
            return 'new_user'
        elif n <= thr_list[1]:
            return 'regular_user'
        else:
            return 'super_user'


    def get_user_status_from_id(self, user_id: int):
        return self._get_status_from_number_of_articles(self.get_nr_of_articles_by_user_id(user_id))


    def get_user_status(self) -> pd.DataFrame:
        """ Classifies all the users and returns a DataFrame
         Examples:
             >>> UserList().get_user_status().user.tolist()[: 3]
             ['regular_user', 'super_user', 'regular_user']
        """
        status = pd.DataFrame(self.user_matrix.sum(axis=1), columns=['read_articles'])
        status['user'] = status['read_articles'].apply(self._get_status_from_number_of_articles)
        return status


# class ArticleList():
#     def __init__(self):
#         self.file = paths.articles_path
#         self.df = pd.read_csv(self.file, index_col=0)
#
