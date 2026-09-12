import os
import struct

def sleb128(value):
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if (value == 0 and (byte & 0x40) == 0) or (value == -1 and (byte & 0x40) != 0):
            result.append(byte)
            break
        result.append(byte | 0x80)
    return result

def uleb128(value):
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value == 0:
            result.append(byte)
            break
        result.append(byte | 0x80)
    return result

# Architecture constants from v16.rs & train.rs
L1 = 1024
FC0_OUT = 32
N_BUCKETS = 8
FC1_IN = 64
FC1_OUT = 32
PSQ_DIMS = 22528
THREAT_DIMS = 59808
PAIR_DIMS = 4560
BIG_INPUT_DIMS = PSQ_DIMS + THREAT_DIMS + PAIR_DIMS

def main():
    if not os.path.exists('quantised.bin'):
        print('Không tìm th?y quantised.bin! Hãy d?m b?o b?n dã train xong trên Kaggle.')
        return
        
    print('Ðang chuy?n d?i quantised.bin sang rudim.nnue chu?n Stockfish LEB128...')
    
    with open('quantised.bin', 'rb') as f_in, open('rudim.nnue', 'wb') as f_out:
        # 1. Version
        f_out.write(struct.pack('<I', 0x7AF32F20))
        # 2. File Hash (Dummy)
        f_out.write(struct.pack('<I', 0))
        # 3. Description
        desc = b'Rudim SFNNv16 Trained on Kaggle'
        f_out.write(struct.pack('<I', len(desc)))
        f_out.write(desc)
        # 4. THash (Dummy)
        f_out.write(struct.pack('<I', 0))
        
        # Read biases (L0) - L1 i16 elements
        l0b_data = f_in.read(L1 * 2)
        l0b = struct.unpack(f'<{L1}h', l0b_data)
        for b in l0b:
            f_out.write(sleb128(b))
            
        # Read weights (L0) - (BIG_INPUT_DIMS, L1) i16 elements
        l0w_data = f_in.read(BIG_INPUT_DIMS * L1 * 2)
        l0w = struct.unpack(f'<{BIG_INPUT_DIMS * L1}h', l0w_data)
        for w in l0w:
            f_out.write(sleb128(w))
            
        # AHash (Dummy)
        f_out.write(struct.pack('<I', 0))
        
        print('Chuy?n d?i thành công! B?n có th? dùng file rudim.nnue cho Engine.')

if __name__ == '__main__':
    # main() # B? comment khi ch?y th?c t? trên Kaggle
    pass
