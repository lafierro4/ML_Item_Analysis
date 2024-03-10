import tkinter as tk
from tkinter import ttk as ttk
import os,metainfo


main_window = tk.Tk()
main_window.geometry("800x600")
main_window.title("Item Analyser")
photo_icon = tk.PhotoImage(file = os.path.join(f"dragontail-{metainfo.VERSION}",metainfo.VERSION,"img","item","1517.png"))
main_window.iconphoto(False, photo_icon)

main_window.mainloop()