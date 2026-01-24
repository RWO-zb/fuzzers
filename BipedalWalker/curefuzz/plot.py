import pickle
import numpy as np

def analyze_crash_lineage(log_file='selection_log.pkl'):
    print(f"Loading {log_file}...")
    try:
        with open(log_file, 'rb') as f:
            log_data = pickle.load(f)
    except FileNotFoundError:
        print("Error: Log file not found.")
        return

    # 1. 构建家谱图 (Child -> Parent 映射)
    # 使用 bytes 作为字典的 key，因为 numpy array 不能直接做 key
    parent_map = {}
    crash_inputs = []

    print("Building lineage graph...")
    for i, entry in enumerate(log_data):
        # 获取父子状态
        parent_state = entry['seed_state']
        child_state = entry['mutate_state']
        is_crash = entry['did_crash']

        # 转换为 bytes 以便哈希去重
        p_bytes = parent_state.tobytes()
        c_bytes = child_state.tobytes()

        # 记录关系：Child 是由 Parent 变异来的
        # 注意：如果同一个 Child 被多次生成，记录任意一个 Parent 即可（通常来源于同一个祖先）
        parent_map[c_bytes] = p_bytes

        if is_crash:
            crash_inputs.append(c_bytes)

    print(f"Total log entries: {len(log_data)}")
    print(f"Total crashes found: {len(crash_inputs)}")

    # 2. 回溯每一个 Crash 到其原始祖先 (Root)
    unique_roots = set()
    
    print("Tracing back lineages...")
    for crash_bytes in crash_inputs:
        curr = crash_bytes
        path_length = 0
        
        # 不断向上寻找父节点，直到找不到父节点为止
        # 找不到父节点意味着它不是在 Fuzzing 循环中生成的，而是初始语料库中的种子
        while curr in parent_map:
            curr = parent_map[curr]
            path_length += 1
            
            # 防止死循环（虽然理论上不会有环）
            if path_length > 10000: 
                print("Warning: lineage too deep or cycle detected.")
                break
        
        # 此时的 curr 就是初始种子 (Root Seed)
        unique_roots.add(curr)

    # 3. 输出结果
    print("-" * 30)
    print(f"Analysis Result:")
    print(f"Number of distinct initial seeds leading to crashes: {len(unique_roots)}")
    print("-" * 30)

if __name__ == "__main__":
    analyze_crash_lineage()