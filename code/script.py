import subprocess

script_path = "UniKP_kcat.py"
log_file = "output.log"
# 使用 subprocess 执行，并将 stdout 和 stderr 重定向到日志文件
with open(log_file, "w") as f:
    process = subprocess.Popen(
        ["python", script_path],
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True
    )
    process.wait()