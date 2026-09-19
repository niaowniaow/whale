import subprocess
import time

def test_time_control(name, wtime, btime, winc=0, binc=0):
    proc = subprocess.Popen(
        [r"C:\Users\newo\Downloads\whale\target\release\whale.exe"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    proc.stdin.write("uci\n")
    proc.stdin.write("isready\n")
    proc.stdin.flush()
    
    while True:
        line = proc.stdout.readline()
        if "readyok" in line:
            break
            
    proc.stdin.write("position startpos\n")
    cmd = f"go wtime {wtime} btime {btime} winc {winc} binc {binc}\n"
    start = time.perf_counter()
    proc.stdin.write(cmd)
    proc.stdin.flush()
    
    best_move = ""
    max_depth = 0
    while True:
        line = proc.stdout.readline()
        if "depth" in line:
            parts = line.strip().split()
            if "depth" in parts:
                idx = parts.index("depth")
                if idx + 1 < len(parts):
                    try:
                        max_depth = max(max_depth, int(parts[idx+1]))
                    except:
                        pass
        if "bestmove" in line:
            best_move = line.strip()
            break
            
    elapsed = (time.perf_counter() - start) * 1000.0
    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.terminate()
    
    print(f"[{name}] wtime={wtime}ms inc={winc}ms -> Depth {max_depth} | Time: {elapsed:.0f}ms | {best_move}")

print("=== TESTING WHALE DYNAMIC TIME MANAGEMENT ===")
test_time_control("Bullet Scramble (5s)", 5000, 5000, 0, 0)
test_time_control("Bullet 1+0 (60s)", 60000, 60000, 0, 0)
test_time_control("Blitz 3+2 (180s)", 180000, 180000, 2000, 2000)
test_time_control("Blitz 5+0 (300s)", 300000, 300000, 0, 0)
test_time_control("Rapid 10+0 (600s)", 600000, 600000, 0, 0)
