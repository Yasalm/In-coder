import abc
from woodwork.logical_types import Unknown
import woodwork as ww
import pandas as pd
from pathlib import Path

"""
qdrant client accepts filters of must, should, must_not
"""


def get_types(df):
    df = pd.DataFrame(df)
    df.ww.init()
    dtypes = df.ww.types
    select_types = lambda row: row['Logical Type'].type_string if not isinstance(row['Logical Type'], Unknown) else row[
        'Physical Type']
    dtypes['Type'] = dtypes.apply(lambda row: select_types(row), axis=1)
    return dtypes.reset_index()[['Column', 'Type']].to_dict(orient='records')


class Schema:
    def __init__(self, data):
        self.data = pd.DataFrame(data)
        self._dtypes = get_types(self.data)
        self.fields = {}  # COLUMN:{Type:type, schema:schema},

    @property
    def types(self):
        if hasattr(self, 'data_types'):
            return self.data_types
        data_types = {}
        for dtype in self._dtypes:
            if isinstance(dtype['Type'], str):
                if self.check_if_list(dtype):
                    data_types[dtype['Column']] = 'list'
                else:
                    data_types[dtype['Column']] = dtype['Type']
            else:
                # print(dir(dtype['Type']))
                data_types[dtype['Column']] = dtype['Type'].type_string
        self.data_types = data_types
        return data_types

    def get_fields(self, ):
        dtypes = self.types
        for column, dtype in dtypes.items():
            if dtype == 'list':
                print('column ', column + ' is a list type', dtype)

    def check_if_list(self, dtype) -> bool:
        column = self.data[dtype['Column']].iloc[0]
        check = column.startswith('[')
        if check:
            return True
        return False
