def print_table(d: dict, limit=9999):
    """Print the given OrderedDict as a table.

    Args:
        d (dict): The dict to print.
        limit (None, optional): The row limit.
    """
    print(
        " ".join(
            [
                key.ljust(_longest_column(d, key, limit) + 3)
                for key in list(d[0].keys())
            ]
        )
    )
    print("-" * 100)
    for row in d[:10]:
        print(
            " ".join(
                [
                    value.ljust(_longest_column(d, key, limit) + 3)
                    for (key, value) in list(row.items())
                ]
            )
        )


def _longest_column(d: dict, column_name: str, limit=9999):
    longest = 0
    for row in d[:limit]:
        if len(row[column_name]) > longest:
            longest = len(row[column_name])
    return longest
