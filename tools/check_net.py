import subprocess
import sys

proc = subprocess.Popen(
    ["target/release/whale.exe"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
    bufsize=1,
)


def send(cmd):
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()


def wait_for(token, timeout=60):
    import time
    start = time.time()
    last = ""
    while time.time() - start < timeout:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        if line.startswith("info ") and ("score" in line or "aprm" in line):
            last = line
            print(line[:160])
        if token in line:
            return last
    return last


send("uci")
wait_for("uciok")
send("setoption name EvalFile value nn-f3d2f6a8b12a.nnue")
send("isready")
wait_for("readyok")
send("position startpos")
send("go depth 8")
wait_for("bestmove", timeout=120)
send("quit")
proc.terminate()
