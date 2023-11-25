import random
import re

import mysql.connector


class KGMatcher:
    def __init__(self):
        self.no_match_entity_id = 0

        self.mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Apple3328823",
            database="kgc"
        )
        self.mycursor = self.mydb.cursor()

    def get_entity_ids(self, input_tokens):
        input_entity_ids = []
        for t in input_tokens:
            pt = self.remove_special_characters(t)
            tid = self.match_token(pt)
            if tid:
                input_entity_ids.append(tid[0])
            else:
                tid = self.match_expose_token(pt)
                if tid is not None:
                    input_entity_ids.append(tid)
                else:
                    input_entity_ids.append(self.no_match_entity_id)

        # for t in input_tokens:
        #     input_entity_ids.append(0)
        return input_entity_ids

    def match_token(self, token):
        # Check if the name exists in the table
        query = "SELECT id FROM entity2id WHERE name = %s"
        self.mycursor.execute(query, (token,))
        result = self.mycursor.fetchone()
        return result

    def match_expose_token(self, token):
        n_l = self.split_camel(token)
        token_ids = []
        for word in n_l:
            tid = self.match_token(word)
            token_ids.append(tid)
        if len(token_ids) > 0:
            return token_ids[len(token_ids)-1]
        else:
            return None


    def split_camel(phrase):
        # Split the phrase based on camel case
        split_phrase = re.findall(r'[A-Z](?:[a-z]+|$)', phrase)
        return split_phrase

    def remove_special_characters(self, input_string):
        # Use a regular expression to remove all non-alphanumeric characters
        return re.sub(r'[^a-zA-Z0-9]', '', input_string)