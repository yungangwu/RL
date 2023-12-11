import numpy as np

def count_consecutive_elements(subarray, player):
    consecutive_count = 0
    max_consecutive_count = 0

    for element in subarray:
        if element == player:
            consecutive_count += 1
            max_consecutive_count = max(max_consecutive_count, consecutive_count)
        else:
            consecutive_count = 0

    return max_consecutive_count

def check_num_in_board(board, player, row, col, num):
    # 获取棋盘的行数和列数
    rows, cols = board.shape

    # 检查行
    row_start = max(0, col - num + 1)
    row_end = min(cols, col + num)
    row_player_num = count_consecutive_elements(board[row, row_start:row_end], player)
    if row_player_num >= num:
        return True

    # 检查列
    col_start = max(0, row - num + 1)
    col_end = min(rows, row + num)
    col_player_num = count_consecutive_elements(board[col_start:col_end, col], player)
    if col_player_num >= num:
        return True

    # 检查主对角线
    diag_start_row = row
    diag_start_col = col
    while diag_start_row > 0 and diag_start_col > 0:
        diag_start_row -= 1
        diag_start_col -= 1

    diag_end_row = row
    diag_end_col = col
    while diag_end_row < rows and diag_end_col < cols:
        diag_end_row += 1
        diag_end_col += 1

    diag = np.diagonal(board[diag_start_row:diag_end_row, diag_start_col:diag_end_col])
    diag_player_num = count_consecutive_elements(diag, player)
    if diag_player_num >= num:
        return True

    # 检查副对角线
    rev_board = np.fliplr(board)
    rev_row = row
    rev_col = cols - 1 - col

    rev_diag_start_row = rev_row
    rev_diag_start_col = rev_col
    while rev_diag_start_row > 0 and rev_diag_start_col > 0:
        rev_diag_start_row -= 1
        rev_diag_start_col -= 1

    rev_diag_end_row = rev_row
    rev_diag_end_col = rev_col
    while rev_diag_end_row < rows and rev_diag_end_col < cols:
        rev_diag_end_row += 1
        rev_diag_end_col += 1

    rev_diag = np.diagonal(rev_board[rev_diag_start_row:rev_diag_end_row, rev_diag_start_col:rev_diag_end_col])
    rev_diag_player_num = count_consecutive_elements(rev_diag, player)
    if rev_diag_player_num >= num:
        return True

    return False


board = [
    [1, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0]
]

board = np.array(board)

row = 4
col = 4
num = 3

if check_num_in_board(board, 1, row, col, num):
    print("存在", num, "子连珠！")
else:
    print("不存在", num, "子连珠。")