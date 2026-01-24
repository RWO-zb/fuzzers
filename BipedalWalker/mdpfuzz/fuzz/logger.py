import os
import numpy as np
import pandas as pd

from typing import Optional, List, Dict, Union


class Logger:
    '''
    A class for logging data values to a .txt file in a specific format.
    '''

    def __init__(self, filepath: str, columns: List[str] = None, delimiter: str = '; ') -> None:
        self.delimiter = delimiter

        if columns is None:
            assert os.path.isfile(filepath)
            with open(filepath, 'r') as file:
                header_line = file.readline().strip()
                columns = header_line.split(self.delimiter)
            # print('No columns provided; found {} columns in the file'.format(len(columns)))

        assert len(columns) > 0
        assert np.all([isinstance(c, str) for c in columns])

        self.filepath = filepath
        self.columns = columns.copy() # type: List[str]
        self.n = len(self.columns)


    def write_columns(self):
        with open(self.filepath, 'w') as file:
            file.write(self.delimiter.join(self.columns) + '\n')


    def log(self, **kwargs) -> None:
        '''Serializes and appends the data to log.'''
        data_serialized = {k: self._serialize(kwargs.get(k, None)) for k in self.columns}
        self._log(data_serialized)


    def load_logs(self) -> pd.DataFrame:
        '''
        Load logs from the file and return as a Pandas DataFrame.
        '''
        data = []
        with open(self.filepath, 'r') as file:
            header_line = file.readline().strip()
            assert header_line.split(self.delimiter) == self.columns, header_line.split(self.delimiter)
            for line in file:
                values = [v.strip() for v in line.strip().split(self.delimiter)]
                assert len(values) == self.n
                data.append([self._deserialize(v) if v != 'None' else None for v in values])

        return pd.DataFrame(data, columns=self.columns)


    def _log(self, data: Dict[str, str]) -> None:
        '''Appends a line to the log file.'''
        log_line = self.delimiter.join([data[k] for k in self.columns])

        with open(self.filepath, 'a') as file:
            file.write(log_line + '\n')


    def _serialize(self, data = None) -> str:
        if data is None:
            return 'None'
        elif isinstance(data, np.ndarray):
            return np.array2string(data, separator=',').replace('\n', '')
        else:
            return str(data)


    def _deserialize(self, data: str) -> Union[np.ndarray, float, bool]:
        if data.startswith('['):
            return np.array(eval('np.array(' + data + ')'))
        elif data.endswith('e'):
            return data == 'True'
        else:
            return float(data)


class FuzzerLogger:

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        # [修改] 新增 RootID 列
        self.columns = [
            'Input', 'Oracle', 'Reward', 'EpisodeLength', 'Sensitivity', 
            'Coverage', 'Generation', 'TestExecTime', 'CoverageTime', 'RunTime',
            'BD_Distance', 'BD_MeanAngle', 'RootID'
        ]
        self.delimiter = '; '

    def log(self,
            input: Optional[np.ndarray] = None,
            oracle: Optional[bool] = None,
            reward: Optional[float] = None,
            episode_length: Optional[int] = None,
            sensitivity: Optional[float] = None,
            coverage: Optional[float] = None,
            Generation: Optional[int] = None,
            run_time: Optional[float] = None,
            test_exec_time: Optional[float] = None,
            coverage_time: Optional[float] = None,
            bd_distance: Optional[float] = None,
            bd_mean_angle: Optional[float] = None,
            # [修改] 新增 root_id 参数
            root_id: Optional[int] = None
        ) -> None:
        '''
        Log values to the file.
        '''
        log_data = {
            'Input': np.array2string(input, separator=',').replace('\n', '') if input is not None else 'None',
            'Oracle': str(oracle) if oracle is not None else 'None',
            'Reward': str(reward) if reward is not None else 'None',
            'EpisodeLength': str(episode_length) if episode_length is not None else 'None',
            'Sensitivity': str(sensitivity) if sensitivity is not None else 'None',
            'Coverage': str(coverage) if coverage is not None else 'None',
            'Generation': str(Generation) if Generation is not None else 'None',
            'RunTime': str(run_time) if run_time is not None else 'None',
            'TestExecTime': str(test_exec_time) if test_exec_time is not None else 'None',
            'CoverageTime': str(coverage_time) if coverage_time is not None else 'None',
            'BD_Distance': str(bd_distance) if bd_distance is not None else 'None',
            'BD_MeanAngle': str(bd_mean_angle) if bd_mean_angle is not None else 'None',
            # [修改] 记录 RootID
            'RootID': str(root_id) if root_id is not None else 'None'
        }
        
        log_line = self.delimiter.join([log_data[k] for k in self.columns])

        with open(self.filepath, 'a') as file:
            file.write(log_line + '\n')


    def load_logs(self) -> pd.DataFrame:
        '''
        Load logs from the file and return as a Pandas DataFrame.
        '''
        data = []
        malformed_lines = []
        with open(self.filepath, 'r') as file:
            header_line = file.readline().strip()
            # 简单校验列数是否大致匹配（兼容旧日志可能报错的情况）
            file_cols = header_line.split(self.delimiter)
            
            for num_line, line in enumerate(file):
                try:
                    values = [v.strip() for v in line.strip().split(self.delimiter)]
                    
                    row_dict = {}
                    # 按照 self.columns 的顺序解析，如果日志列数少于代码定义列数（比如旧日志），则填充 None
                    for i, col in enumerate(self.columns):
                        if i < len(values):
                            val_str = values[i]
                        else:
                            val_str = 'None'

                        if col == 'Input':
                             val = np.array(eval('np.array(' + val_str + ')')) if val_str != 'None' else None
                        elif col == 'Oracle':
                             val = val_str == 'True' if val_str != 'None' else None
                        elif col in ['EpisodeLength', 'Generation', 'RootID']:
                             val = int(float(val_str)) if val_str != 'None' else None
                        else:
                             val = float(val_str) if val_str != 'None' else None
                        row_dict[col] = val
                    
                    data.append([row_dict[c] for c in self.columns])
                except Exception as e:
                    malformed_lines.append(f'\tLine {num_line}: Error: {e}')
        
        return pd.DataFrame(data, columns=self.columns)


    def write_columns(self):
        with open(self.filepath, 'w') as file:
            file.write(self.delimiter.join(self.columns) + '\n')