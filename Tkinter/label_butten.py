import tkinter as tk

window = tk.Tk()
window.title('my window')
window.geometry('200x100')


l = tk.Label(window,
    text='OMG! this is TK',
    bg='green',
    font=('Arial', 12),
    width=15, height=2
    )
l.pack()


on_hit = False
def hit_me():
    global on_hit
    if on_hit == False:
        on_hit = True
        var.set('you hit me')
    else:
        on_hit = False
        var.set('')

var = tk.StringVar()  # 文字变量存储器
l = tk.Label(window,
    textvariable=var, # 这个变量，可以变化
    bg='green', font=('Arial', 12), width=15, height=2
    )
l.pack()

b = tk.Button(window,
    text='hit me',
    width=15, height=2,
    command=hit_me
    )
b.pack()


window.mainloop()
