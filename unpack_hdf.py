import threading
import time

class HDFThread(threading.Thread):
    """

    """
    def __init__(self, thread_id, name, in_path, out_path, file_name):
        self.thread_id = thread_id
        self.name = name
        self.in_path = in_path
        self.out_path = out_path
        self.file_name = file_name

    def run(self):
        print("Starting thread id: ", self.thread_id, " called: ", self.name)





if __name__ == "__main__":



    print("PROGRAMM ENDE")