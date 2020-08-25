import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class TicTacToe(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 64)
        self.l2 = nn.Linear(64, 36)
        self.out = nn.Linear(36, 9)

    def forward(self, input):
        input = input.float()
        input = self.l1(input)
        input = torch.relu(input)
        input = self.l2(input)
        input = torch.relu(input)
        input = self.out(input)
        return torch.sigmoid(input)

moves = np.array([ 0b100000000,
                            0b010000000,
                            0b001000000,
                            0b000100000,
                            0b000010000,
                            0b000001000,
                            0b000000100,
                            0b000000010,
                            0b000000001])


def convert_to_tensor(board):
    return torch.from_numpy(np.int16(board))

def convert_to_move(tensor):
    idx = torch.argmax((tensor+1) * torch.tensor(bit_array(np.bitwise_xor(sum(bitboard),0b111111111)))).item()

    return moves[idx],idx

def bit_array(b):
    b = np.uint16(b)
    string = str(bin(b))[2:]
    string = (9-len(string))*'0' + string
    string = string[0:9]
    return np.array([int(d) for d in string])

bitboard = np.array([np.uint16(0),np.uint16(0)])
turn = np.bool_(0)

def bit_visual(bb,sqr):
    string = str(bin(bb))[2:]
    string = ((sqr*sqr)-len(string))*'0' + string
    for i in range(sqr):
        print(string[sqr*i:sqr*(i+1)])

def visualize():
    X = str(bin(bitboard[0]))[2:]
    O = str(bin(bitboard[1]))[2:]
    X = (9-len(X))*'0' + X
    O = (9-len(O))*'0' + O
    output = ""
    for i in range(9):
        if X[i] == '1':
            output += 'x '
        elif O[i] == '1':
            output += 'o '
        else :
            output += '□ '
    print(output[0:6])
    print(output[6:12])
    print(output[12:])
    print("")


def move(player,square): # TWO IDENTICAL MOVES CAUSES UNDO
    global turn
    if not isinstance(square, np.int32) : 
        square = square[0]
    if np.bitwise_or(np.bitwise_and(np.sum(bitboard),square), np.bitwise_xor(player,turn)) : 
        print("invalid")
    else : 
        bitboard[player] += square
        turn = np.bitwise_not(turn)
        

win_conditions = np.array([ 0b111000000,
                            0b000111000,
                            0b000000111,
                            0b100100100,
                            0b010010010,
                            0b001001001,
                            0b100010001,
                            0b001010100])

win_conditions = np.uint16(win_conditions)

def win():
    global turn
    for i in win_conditions:
        if np.bitwise_xor(np.bitwise_and(bitboard[np.uint8(np.bitwise_not(turn))],i),i) == 0: return turn

convert = {
    '1' : np.int32(0x100),
    '2' : np.int32(0x80),
    '3' : np.int32(0x40),
    '4' : np.int32(0x20),
    '5' : np.int32(0x10),
    '6' : np.int32(0x8),
    '7' : np.int32(0x4),
    '8' : np.int32(0x2),
    '9' : np.int32(0x1)
}
def idx_to_tensor(i):
    out = torch.tensor([0,0,0,0,0,0,0,0,0],dtype=torch.bool)
    out[i] = 1
    return out

def AiDuel(model):
    inputs = []
    outputs = []
    for _ in range(4):
        inputs.append(convert_to_tensor(bitboard))
        m,i = convert_to_move(model(inputs[-1]))
        move(0,m)
        outputs.append(idx_to_tensor(i))
        if win() is not None: return 1,inputs,outputs
        move(1,np.random.default_rng().choice(moves[bit_array(np.bitwise_xor(sum(bitboard),0b111111111)) == 1]))
        if win() is not None: return 0,inputs,outputs
    move(0,moves[bit_array(np.bitwise_xor(sum(bitboard),0b111111111)) == 1])
    if win() is not None: return 1,inputs,outputs
    return -1,inputs,outputs

def play_x(model):
    reset()
    while(True):
        for _ in range(4):
            move(0,convert_to_move(model(convert_to_tensor(bitboard))))
            visualize()
            if win() is not None: break
            x = input()
            if len(x) != 1: break
            if int(x) <= 0: break
            if int(x) > 9: break
            move(1,convert[x])
            visualize()
            if win() is not None: break
        print("-----------------------------------------------------")
        print("New Game!")
        reset()

def reset():
    global bitboard
    global turn
    bitboard = np.array([np.uint16(0),np.uint16(0)])
    turn = np.bool_(0)

torch.autograd.set_detect_anomaly(True)
def train(num_games,model):
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    win = 0
    los = 0
    tie = 0
    owin = 0
    olos = 0
    otie = 0
    win_arr = []
    loss_arr = []
    draw_arr = []
    for i in range(num_games):
        reset()
        result,inputs,outputs = AiDuel(model)
        if result == 1:
            win+=1
            owin+=1
            for j in range(len(inputs)):
                optimizer.zero_grad()
                loss = nn.MSELoss()
                output = loss(model(inputs[j]),outputs[j].float() *0.5 + 0.5)
                output.backward()
                optimizer.step()
        elif result == 0:
            los+=1
            olos+=1
            for j in range(len(inputs)):
                optimizer.zero_grad()
                loss = nn.MSELoss()
                output = loss(model(inputs[j]),(~outputs[j]).float() *0.5)
                output.backward()
                optimizer.step()
        else : 
            tie+=1
            otie+=1
        if (i % 1000 == 0):
            if i == 0 : continue
            print(100*win/1000,"% WIN")
            print(100*los/1000,"% LOSS")
            print(100*tie/1000,"% TIE")
            print("")
            print(100*owin/(i+1),"% WIN OVERALL")
            print(100*olos/(i+1),"% LOSS OVERALL")
            print(100*otie/(i+1),"% TIE OVERALL")
            win_arr.append(100*win/1000)
            loss_arr.append(100*los/1000)
            draw_arr.append(100*tie/1000)
            plt.cla()
            plt.ylabel("Accuracies")
            plt.xlabel("1000 Games Played")
            plt.plot(win_arr,label='Win')
            plt.plot(loss_arr,label='Loss')
            plt.plot(draw_arr,label='Draw')
            plt.legend()
            plt.savefig("results")
            win=0
            tie=0
            los=0
            print("")
    return model
model = TicTacToe()
model = train(1000000,model)

play_x(model)