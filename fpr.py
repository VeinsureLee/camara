import subprocess
import os
import platform
from typing import Union, Optional


def run_frpc(
    frp_dir: str = r"frp_0.52.3_windows_amd64",
    config_file: str = "frpc.toml",
    hide_window: bool = False,
    timeout: Optional[float] = None
) -> subprocess.CompletedProcess:
    """
    运行 frpc 客户端的封装函数

    :param frp_dir: frp目录路径（默认：当前目录下的 frp_0.52.3_windows_amd64）
    :param config_file: 配置文件名称（默认：frpc.toml）
    :param hide_window: 是否隐藏控制台窗口（仅Windows有效）
    :param timeout: 命令超时时间（秒）
    :return: subprocess.CompletedProcess 对象
    :raises FileNotFoundError: 当文件不存在时
    :raises subprocess.CalledProcessError: 当命令执行失败时
    """
    # 自动确定默认路径
    if frp_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        frp_dir = os.path.join(base_dir, "frp_0.52.3_windows_amd64")

    # 构建完整路径
    exe_path = os.path.join(frp_dir, "frpc.exe")
    config_path = os.path.join(frp_dir, config_file)

    # 校验文件存在性
    if not os.path.isfile(exe_path):
        raise FileNotFoundError(f"frpc.exe 不存在于路径: {exe_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"配置文件 {config_file} 不存在于路径: {config_path}")

    # 构建命令参数
    cmd = [exe_path, "-c", config_path]

    # Windows隐藏窗口标志
    kwargs = {}
    if hide_window and platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    # 执行命令
    try:
        return subprocess.run(
            cmd,
            check=True,
            timeout=timeout,
            **kwargs
        )
    except subprocess.TimeoutExpired as e:
        print(f"命令执行超时（{timeout}秒）")
        raise
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，返回码: {e.returncode}")
        raise


def main():
    try:
        # 基础用法
        result = run_frpc()
        print("命令执行成功")

        # 高级用法示例
        # result = run_frpc(
        #     frp_dir=r"C:\custom\frp_directory",
        #     config_file="custom_config.toml",
        #     hide_window=True,
        #     timeout=30
        # )
    except FileNotFoundError as e:
        print(f"文件未找到错误: {str(e)}")
    except subprocess.CalledProcessError as e:
        print(f"命令执行错误: {str(e)}")
    except Exception as e:
        print(f"未知错误: {str(e)}")


# 使用示例
if __name__ == "__main__":
    main()
