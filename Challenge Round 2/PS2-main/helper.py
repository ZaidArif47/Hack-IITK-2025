
import ast

def convert_to_list(url_str):
    try:
        # Convert the string representation of a list to an actual list
        url_list = ast.literal_eval(url_str)
        if isinstance(url_list, list):
            return url_list
        else:
            return []
    except (ValueError, SyntaxError):
        # In case the string is not a valid list format
        return []