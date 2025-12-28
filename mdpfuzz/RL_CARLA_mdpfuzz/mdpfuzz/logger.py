import os
import numpy as np
import pandas as pd

from typing import Optional, List, Dict, Union


class Logger:
    '''
    A class for logging data values to a .txt file in a specific format.

    Args:
    - filepath (str): The path to the log file.

    Usage:
    - Initialize the logger with a file path.
    - Write the header with the write_columns method before logging.
    - Use the log method to add a new log entry.
    - Use load_logs to load logs into a Pandas DataFrame.
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

        Returns:
        - pd.DataFrame: DataFrame containing the logged data.
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
        self.columns = ['input', 'oracle', 'reward', 'episode_length', 'sensitivity', 'coverage', 'test_exec_time', 'coverage_time', 'run_time']
        self.delimiter = '; '

    def log(self,
            input: Optional[np.ndarray] = None,
            oracle: Optional[bool] = None,
            reward: Optional[float] = None,
            episode_length: Optional[int] = None,
            sensitivity: Optional[float] = None,
            coverage: Optional[float] = None,
            run_time: Optional[float] = None,
            test_exec_time: Optional[float] = None,
            coverage_time: Optional[float] = None
        ) -> None:
        '''
        Log values to the file.

        Args:
        - input (Optional[np.ndarray]): Input value as a NumPy array.
        - oracle (Optional[bool]): Oracle value as a boolean.
        - reward (Optional[float]): Reward value as a floating-point number.
        - episode_length (Optional[int]): Episode length (positive integer).
        - sensitivity (Optional[float]): Sensitivity value as a floating-point number.
        - coverage (Optional[float]): Coverage value as a floating-point number.
        - run_time (Optional[float]): Run time value as a floating-point number.
        - test_exec_time (Optional[float]): Test execution time value as a floating-point number.
        - coverage_time (Optional[float]): Time to compute a coverage value as a floating-point number.
        '''
        log_data = {
            #TODO: compared to the pool np.savetxt(.), np.array2string is less accurate
            # It would be problematic if the precision difference causes reproducibility issue...
            'input': np.array2string(input, separator=',').replace('\n', '') if input is not None else 'None',
            'oracle': str(oracle) if oracle is not None else 'None',
            'reward': str(reward) if reward is not None else 'None',
            'episode_length': str(episode_length) if episode_length is not None else 'None',
            'sensitivity': str(sensitivity) if sensitivity is not None else 'None',
            'coverage': str(coverage) if coverage is not None else 'None',
            'run_time': str(run_time) if run_time is not None else 'None',
            'test_exec_time': str(test_exec_time) if test_exec_time is not None else 'None',
            'coverage_time': str(coverage_time) if coverage_time is not None else 'None'
        }

        # ensures correct ordering by using the columns (weakness found when Python version is 3.5)
        log_line = self.delimiter.join([log_data[k] for k in self.columns])

        with open(self.filepath, 'a') as file:
            file.write(log_line + '\n')


    def load_logs(self) -> pd.DataFrame:
        '''
        Load logs from the file and return as a Pandas DataFrame.

        Returns:
        - pd.DataFrame: DataFrame containing the logged data.
        '''
        data = []
        malformed_lines = []
        with open(self.filepath, 'r') as file:
            header_line = file.readline().strip()
            assert header_line.split(self.delimiter) == self.columns, header_line.split(self.delimiter)
            for num_line, line in enumerate(file):
                try:
                    values = [v.strip() for v in line.strip().split(self.delimiter)]
                    input = np.array(eval('np.array(' + values[0] + ')')) if values[0] != 'None' else None
                    oracle = values[1] == 'True' if values[1] != 'None' else None
                    reward = float(values[2]) if values[2] != 'None' else None
                    episode_length = int(values[3]) if values[3] != 'None' else None
                    sensitivity = float(values[4]) if values[4] != 'None' else None
                    coverage = float(values[5]) if values[5] != 'None' else None
                    test_exec_time = float(values[6]) if values[6] != 'None' else None
                    coverage_time = float(values[7]) if values[7] != 'None' else None
                    run_time = float(values[8]) if values[8] != 'None' else None

                    data.append([input, oracle, reward, episode_length, sensitivity, coverage, test_exec_time, coverage_time, run_time])
                except:
                    malformed_lines.append('\tLine {}: "{}"'.format(num_line, line.strip()))
        # if malformed_lines != []:
        #     print('Found malformed lines in file "{}"'.format(self.filepath))
        #     for info in malformed_lines:
        #         print(info)
        return pd.DataFrame(data, columns=self.columns)


    def write_columns(self):
        with open(self.filepath, 'w') as file:
            file.write(self.delimiter.join(self.columns) + '\n')


if __name__ == '__main__':
    log_file_example_path = 'log_file_example.txt'
    logger = FuzzerLogger(log_file_example_path)
    df = logger.load_logs()
    print(df.head(10))