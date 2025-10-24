"""
清理所有实验输出结果
Clean all experimental output files
"""

import os
import shutil
import glob


def clean_results():
    """清理所有结果文件"""
    
    print("="*80)
    print("清理实验结果")
    print("Cleaning Experimental Results")
    print("="*80)
    
    # 要删除的目录
    dirs_to_clean = [
        'results/task1',
        'results/task2/subtask1',
        'results/task2/subtask2',
        'results/task2/subtask3',
        'results/task2'
    ]
    
    # 要删除的文件模式
    files_to_clean = [
        'task1_output.txt',
        'task2_output.txt',
        'task2_output_v2.txt',
        '*.pyc',
        '__pycache__'
    ]
    
    deleted_count = 0
    
    # 清理目录
    print("\n清理结果目录...")
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            try:
                # 统计文件数
                file_count = 0
                for root, dirs, files in os.walk(dir_path):
                    file_count += len(files)
                
                # 删除目录
                shutil.rmtree(dir_path)
                print(f"  ✓ 删除目录: {dir_path} ({file_count} 个文件)")
                deleted_count += file_count
            except Exception as e:
                print(f"  ✗ 删除失败: {dir_path} - {e}")
        else:
            print(f"  - 目录不存在: {dir_path}")
    
    # 清理单个文件
    print("\n清理输出文件...")
    for pattern in files_to_clean:
        matches = glob.glob(pattern, recursive=True)
        for file_path in matches:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"  ✓ 删除文件: {file_path}")
                    deleted_count += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"  ✓ 删除目录: {file_path}")
            except Exception as e:
                print(f"  ✗ 删除失败: {file_path} - {e}")
    
    # 清理Python缓存
    print("\n清理Python缓存...")
    for root, dirs, files in os.walk('.'):
        # 删除 __pycache__ 目录
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                print(f"  ✓ 删除缓存: {pycache_path}")
            except Exception as e:
                print(f"  ✗ 删除失败: {pycache_path} - {e}")
        
        # 删除 .pyc 文件
        for file in files:
            if file.endswith('.pyc'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"  ✓ 删除: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  ✗ 删除失败: {file_path} - {e}")
    
    # 重新创建空的结果目录
    print("\n重新创建结果目录...")
    os.makedirs('results/task1', exist_ok=True)
    os.makedirs('results/task2/subtask1', exist_ok=True)
    os.makedirs('results/task2/subtask2', exist_ok=True)
    os.makedirs('results/task2/subtask3', exist_ok=True)
    print("  ✓ 结果目录已重建")
    
    print("\n" + "="*80)
    print(f"清理完成！共删除 {deleted_count} 个文件/目录")
    print("Clean completed!")
    print("="*80)
    
    # 显示当前状态
    print("\n当前状态:")
    print("  results/task1/ - 空")
    print("  results/task2/subtask1/ - 空")
    print("  results/task2/subtask2/ - 空")
    print("  results/task2/subtask3/ - 空")
    print("\n可以重新运行实验了！")


if __name__ == "__main__":
    # 确认操作
    print("\n⚠️  警告：此操作将删除所有实验结果！")
    print("  - 所有图像文件 (.png)")
    print("  - 所有输出文本 (task*_output.txt)")
    print("  - Python缓存 (__pycache__, *.pyc)")
    
    response = input("\n确定要清理吗？(y/n): ")
    
    if response.lower() in ['y', 'yes', 'Y', 'YES']:
        clean_results()
    else:
        print("\n操作已取消。")

