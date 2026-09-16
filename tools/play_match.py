import subprocess
import time
import sys
import chess

def start_engine(eval_file):
    proc = subprocess.Popen(
        [r".\target\release\rudim.exe"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )
    send_cmd(proc, "uci")
    send_cmd(proc, f"setoption name EvalFile value {eval_file}")
    send_cmd(proc, "setoption name Hash value 256")
    send_cmd(proc, "isready")
    wait_for(proc, "readyok")
    return proc

def send_cmd(proc, cmd):
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()

def wait_for(proc, target):
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        if target in line:
            return line

def get_best_move(proc, moves_str, movetime=500):
    if moves_str:
        send_cmd(proc, f"position startpos moves {moves_str}")
    else:
        send_cmd(proc, "position startpos")
    send_cmd(proc, f"go movetime {movetime}")
    bestmove = None
    score_info = ""
    depth_info = ""
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        if "score" in line and "depth" in line:
            parts = line.split()
            if "score" in parts:
                idx = parts.index("score")
                score_info = " ".join(parts[idx:idx+3])
            if "depth" in parts:
                d_idx = parts.index("depth")
                depth_info = parts[d_idx+1]
        if line.startswith("bestmove"):
            parts = line.split()
            bestmove = parts[1]
            break
    return bestmove, score_info, depth_info

def play_one_game(game_id, white_is_father, movetime=500, max_moves=150):
    father_file = "rudim_farseer_final.nnue"
    son_file = "rudim_farseerT76.nnue"

    proc_father = start_engine(father_file)
    proc_son = start_engine(son_file)

    white_proc = proc_father if white_is_father else proc_son
    black_proc = proc_son if white_is_father else proc_father

    white_name = "Bo (1.5 Ty - Final)" if white_is_father else "Con (400M - Old)"
    black_name = "Con (400M - Old)" if white_is_father else "Bo (1.5 Ty - Final)"

    print("\n" + "="*65)
    print(f"VAN {game_id}: [TRANG] {white_name}  VS  [DEN] {black_name}")
    print(f"Thoi gian: {movetime}ms / nuoc | Gioi han: {max_moves} nuoc")
    print("="*65 + "\n")

    board = chess.Board()
    moves_list = []
    result = None

    for move_num in range(1, max_moves + 1):
        moves_str = " ".join(moves_list)
        move_w, score_w, depth_w = get_best_move(white_proc, moves_str, movetime)
        if not move_w or move_w == "(none)" or move_w == "0000":
            result = "0-1"
            print(f"\nTrang het nuoc di! Den thang.")
            break
        
        try:
            parsed_w = chess.Move.from_uci(move_w)
            board.push(parsed_w)
            moves_list.append(move_w)
        except Exception:
            result = "0-1"
            break

        print(f"{move_num}. {move_w} (d{depth_w}, {score_w})", end=" ", flush=True)

        if board.is_checkmate():
            result = "1-0"
            print(f"\nTrang chieu bi! Trang thang.")
            break
        if board.is_stalemate() or board.can_claim_threefold_repetition() or board.is_insufficient_material() or board.can_claim_fifty_moves():
            result = "1/2-1/2"
            print(f"\nHoa co (luat co vua)!")
            break

        moves_str = " ".join(moves_list)
        move_b, score_b, depth_b = get_best_move(black_proc, moves_str, movetime)
        if not move_b or move_b == "(none)" or move_b == "0000":
            result = "1-0"
            print(f"\nDen het nuoc di! Trang thang.")
            break

        try:
            parsed_b = chess.Move.from_uci(move_b)
            board.push(parsed_b)
            moves_list.append(move_b)
        except Exception:
            result = "1-0"
            break

        print(f"{move_b} (d{depth_b}, {score_b})", flush=True)

        if board.is_checkmate():
            result = "0-1"
            print(f"\nDen chieu bi! Den thang.")
            break
        if board.is_stalemate() or board.can_claim_threefold_repetition() or board.is_insufficient_material() or board.can_claim_fifty_moves():
            result = "1/2-1/2"
            print(f"\nHoa co (luat co vua)!")
            break

    if result is None:
        result = "1/2-1/2"

    send_cmd(proc_father, "quit")
    send_cmd(proc_son, "quit")
    proc_father.terminate()
    proc_son.terminate()

    print(f"\n>>> KET THUC VAN {game_id}: {result} sau {len(moves_list)} nuoc.")
    return result

def main():
    movetime = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    
    score_father = 0.0
    score_son = 0.0

    res1 = play_one_game(1, white_is_father=True, movetime=movetime, max_moves=150)
    if res1 == "1-0":
        score_father += 1.0
    elif res1 == "0-1":
        score_son += 1.0
    else:
        score_father += 0.5
        score_son += 0.5

    res2 = play_one_game(2, white_is_father=False, movetime=movetime, max_moves=150)
    if res2 == "1-0":
        score_son += 1.0
    elif res2 == "0-1":
        score_father += 1.0
    else:
        score_father += 0.5
        score_son += 0.5

    print("\n" + "#"*65)
    print("TONG KET LOAT TRAN (DERBY 2 VAN):")
    print(f"BO  (1.5 Ty - Final): {score_father} diem")
    print(f"CON (400M  - Old)  : {score_son} diem")
    if score_father > score_son:
        print("KET QUA: BO (1.5 Ty) CHIEN THANG AP DAO!")
    elif score_son > score_father:
        print("KET QUA: CON (400M) CHIEN THANG!")
    else:
        print("KET QUA: HAI BEN BAT PHAN THANG BAI (HOA)!")
    print("#"*65 + "\n")

if __name__ == "__main__":
    main()
