import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import os


class DLMBApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("Skin-Lesion-Generator")
        window_width = int(self.winfo_screenwidth() / 2)
        window_height = int(self.winfo_screenheight() / 2)

        self.geometry("{}x{}".format(window_width, window_height))

        position_right = int(window_width - window_width / 2)
        position_down = int(window_height - window_height / 2)

        # Positions the window in the center of the page.
        self.geometry("+{}+{}".format(position_right, position_down))

        # we stack the individual frames on top of each other
        # to be able to lift the corresponding frame when
        # it should be rendered
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        # container.grid_rowconfigure(0, weight=1)
        # container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(0, minsize=500)
        container.grid_rowconfigure(0, minsize=500)

        self.frames = {}
        for PageFrame in (StartPage, ResultsPage, StartTrainingPage):
            page_name = PageFrame.__name__
            frame = PageFrame(parent=container, controller=self)
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.grid_columnconfigure(0, minsize=500)
        self.grid_rowconfigure(0, minsize=500)
        left_frame = tk.Frame(self)
        right_frame = tk.Frame(self)

        left_frame.pack(ipadx=10, ipady=10, side="left", expand=True, fill="both")
        right_frame.pack(ipadx=10, ipady=10, expand=True, fill="both")

        self.checkpoint_list = ttk.Treeview(left_frame)
        self.checkpoint_list.bind('<ButtonRelease-1>', func=self.select_checkpoint)
        self.checkpoint_list.column("#0", width=270, minwidth=270, stretch=tk.NO)

        self.checkpoint_list.heading("#0", text="Run", anchor=tk.W)

        path = "./runs"

        for root, directories, files in os.walk(path, topdown=False):
            for name in directories:
                folder_split = name.split("_")[0:2]
                folder_name = folder_split[0] + "_" + folder_split[1]
                folder_path = os.path.join(root, name)
                folder_elements = os.listdir(folder_path)
                if len(folder_elements) != 1:
                    # want to skip if no elements are present
                    folder = self.checkpoint_list.insert("", index=1, text=folder_name)

                    for file in folder_elements:
                        if "events" not in file:
                            self.checkpoint_list.insert(folder, index="end", values=(folder_path + "/" + file),
                                                        text=file)

        self.checkpoint_list.pack()

        start_training_btn = tk.Button(right_frame, text="Start New Training",
                                       command=lambda: self.handle_training(new=True, checkpoint=None))

        selected_checkpoint = None
        continue_training_btn = tk.Button(right_frame, text="Continue From Checkpoint",
                                          command=lambda: self.handle_training(new=False,
                                                                               checkpoint=selected_checkpoint))

        load_model_btn = tk.Button(right_frame, text="Evaluate Model",
                                   command=lambda: self.handle_model_evaluation(None))
        start_training_btn.pack()
        continue_training_btn.pack()
        load_model_btn.pack()

    def select_checkpoint(self, a):
        cur_item = self.checkpoint_list.focus()
        values = self.checkpoint_list.item(cur_item)['values']
        print(values)
        if len(values) == 0:
            self.checkpoint_list.selection_remove(cur_item)
        else:
            self.controller.selected_checkpoint = values[0]

    def handle_training(self, new, checkpoint):
        if new:
            self.controller.show_frame("StartTrainingPage")
        elif checkpoint is None:
            messagebox.showwarning(title="Invalid selection", message="You need to select a checkpoint in order to "
                                                                      "continue Training!")
        else:
            self.controller.selected_checkpoint = checkpoint
            self.controller.show_frame("StartTrainingPage")
        print("start training")

    def handle_model_evaluation(self, checkpoint):
        if checkpoint is None:
            messagebox.showwarning(title="Invalid selection", message="You need to select a checkpoint in order to "
                                                                      "show results!")
        else:
            self.controller.loaded_checkpoints = checkpoint
            self.controller.show_frame("ResultsPage")

        print("start training")


class ResultsPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Results")
        label.pack(side="top", fill="x", pady=10)
        button = tk.Button(self, text="Go back",
                           command=lambda: controller.show_frame("StartPage"))
        button.pack()


class StartTrainingPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="This is page 2")
        label.pack(side="top", fill="x", pady=10)
        button = tk.Button(self, text="Go to the start page",
                           command=lambda: controller.show_frame("StartPage"))
        button.pack()


if __name__ == "__main__":
    app = DLMBApp()
    app.mainloop()
