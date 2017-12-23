import subprocess
import time

__all__ = ['gpu_using_least_memory']

def filter_alphanumeric(string):
    return ''.join([c for c in string if c.isnumeric()])

def find_min(rows, index):
    min_row = 0
    min_val = float(rows[min_row][index])

    for row_num in range(1, len(rows)):
        row_val = float(rows[row_num][index])
        if row_val < min_val:
            min_val = row_val
            min_row = row_num
    return min_row, min_val


def read_csv(string, return_header=True, remove_nonalpha=True):
    if isinstance(string, bytes):
        string = string.decode('utf-8')
    rows = string.split('\n')
    rows = [row.split(',') for row in rows if row != '']
    if remove_nonalpha:
        rows = [[filter_alphanumeric(entry) for entry in row] for row in rows]
    else:
        rows = [[entry.strip() for entry in row] for row in rows]
    rows = list(map(tuple, rows))
    if not return_header:
        rows = rows[1:]
    return rows

def gpu_using_least_memory():
    """Return id of GPU with lowest memory utilization.
    """
    output = subprocess.check_output(
            "nvidia-smi --query-gpu=index,memory.used --format=csv", shell=True)
    csv = read_csv(output, return_header=False)
    gpu_index, memory_index = 0, 1
    min_row, min_memory = find_min(csv, index=memory_index)
    gpu_id = int(csv[min_row][gpu_index])
    print('GPU {} is using the least memory ({}MiB).'.format(gpu_id, min_memory))
    return gpu_id

