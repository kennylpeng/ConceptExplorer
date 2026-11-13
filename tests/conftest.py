"""Read result dictionaries as frames for numerical assertions."""
import pandas as pd


def frame(output):
    assert type(output) is dict
    assert set(output) == {'data', 'metadata'}
    assert all(isinstance(column, list) for column in output['data'].values())
    assert not {'rank_bound', 'separation_LCB'} & output['data'].keys()
    table = pd.DataFrame(output['data'])
    table.attrs.update(output['metadata'])
    return table


def categories(output):
    return {category: frame(value) for category, value in output.items()}
