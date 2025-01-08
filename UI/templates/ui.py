import tkinter as tk
from tkinter import messagebox,ttk
from PIL import ImageTk, Image
import cv2
import time

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance App")
        self.root.geometry("800x600")
        self.root.configure(bg="#f7fafc")  # bg-gray-100

        self.frames = []

        
        
        # Thêm các frame khác nếu cần
        # Header
        self.header = tk.Frame(self.root, bg="#3b82f6", height=50)
        self.header.pack(fill=tk.X)

        self.logo = tk.Label(self.header, bg="#3b82f6")
        self.logo_img = ImageTk.PhotoImage(file="D:\\Graduation_Project\\image\\ui\\logo.png") 
        self.logo.pack(side=tk.LEFT, padx=10)

        self.menu_button = tk.Button(self.header, text="Menu", command=self.toggle_menu, bg="#3b82f6", fg="white")
        self.menu_button.pack(side=tk.LEFT, padx=10)

        # Menu
        self.menu_frame = tk.Frame(self.root, bg="#1f2937", width=200)
        self.menu_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.menu_frame.pack_propagate(False)  # Không cho phép thay đổi kích thước
        self.menu_frame.pack_forget()
        self.menu_items = [
            "Trang chủ",
            "Đăng kí lớp",
            "Đăng kí sinh viên",
            "Danh sách điểm danh",
            "Điểm danh",
            "Báo cáo",
            "Xuất"
        ]

        for item in self.menu_items:
            btn = tk.Button(self.menu_frame, text=item, command=lambda i=item: self.menu_action(i), bg="#1f2937", fg="white")
            btn.pack(fill=tk.X, padx=5, pady=5)

        
        ################################################################################################
        # Diem danh
        # attendance_frame_diemdanh

        # Tạo khung cho camera
        self.camera_frame = tk.Frame(self.root)
        self.camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ######

        self.attendance_frame_diemdanh = tk.Frame(self.root, bg="white")
        self.attendance_frame_diemdanh.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.attendance_label = tk.Label(self.attendance_frame_diemdanh, text="Thông tin điểm danh", font=("Roboto", 16))
        self.attendance_label.pack(pady=10)

        self.info_frame = tk.Frame(self.attendance_frame_diemdanh, bg="white")
        self.info_frame.pack(pady=10)

        self.class_name_label = tk.Label(self.info_frame, text="Tên lớp: Lớp A", font=("Roboto", 12))
        self.class_name_label.grid(row=0, column=0, sticky="w")

        self.present_count_label = tk.Label(self.info_frame, text="Số lượng có mặt: 20", font=("Roboto", 12))
        self.present_count_label.grid(row=1, column=0, sticky="w")

        self.absent_count_label = tk.Label(self.info_frame, text="Số lượng vắng mặt: 5", font=("Roboto", 12))
        self.absent_count_label.grid(row=2, column=0, sticky="w")

        self.student_info_label = tk.Label(self.attendance_frame_diemdanh, text="Thông tin sinh viên", font=("Roboto", 14, "bold"))
        self.student_info_label.pack(pady=10)

        self.student_name_label = tk.Label(self.attendance_frame_diemdanh, text="Tên: Nguyễn Văn A", font=("Roboto", 12))
        self.student_name_label.pack()

        self.student_id_label = tk.Label(self.attendance_frame_diemdanh, text="ID: 123456", font=("Roboto", 12))
        self.student_id_label.pack()

        self.class_activity_label = tk.Label(self.attendance_frame_diemdanh, text="Lớp sinh hoạt: Lớp B", font=("Roboto", 12))
        self.class_activity_label.pack()

        self.absent_number_label = tk.Label(self.attendance_frame_diemdanh, text="Số vắng: 2", font=("Roboto", 12))
        self.absent_number_label.pack()


         # Khởi tạo camera
        self.video_source = 0  # 0 là camera mặc định
        self.vid = None

        self.canvas = tk.Canvas(self.camera_frame)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        
        ####

        self.attendance_frame_diemdanh.pack_forget()  # Ẩn attendance frame ban đầu
        self.camera_frame.pack_forget()
        ################################################################

        ##############################################################################################
        # Xuat
        # export_frame
        self.export_frame = tk.Frame(self.root, bg="white")
        self.export_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.attendance_label = tk.Label(self.export_frame, text="Thông tin điểm danh", font=("Roboto", 16))
        self.attendance_label.pack(pady=10)

        self.info_frame = tk.Frame(self.export_frame, bg="white")
        self.info_frame.pack(pady=10)

        self.class_name_label = tk.Label(self.info_frame, text="Tên lớp: Lớp A", font=("Roboto", 12))
        self.class_name_label.grid(row=0, column=0, sticky="w")

        self.present_count_label = tk.Label(self.info_frame, text="Số lượng có mặt: 20", font=("Roboto", 12))
        self.present_count_label.grid(row=1, column=0, sticky="w")

        self.absent_count_label = tk.Label(self.info_frame, text="Số lượng vắng mặt: 5", font=("Roboto", 12))
        self.absent_count_label.grid(row=2, column=0, sticky="w")

        self.student_info_label = tk.Label(self.export_frame, text="Thông tin sinh viên", font=("Roboto", 14, "bold"))
        self.student_info_label.pack(pady=10)

        self.student_name_label = tk.Label(self.export_frame, text="Tên: Nguyễn Văn A", font=("Roboto", 12))
        self.student_name_label.pack()

        self.student_id_label = tk.Label(self.export_frame, text="ID: 123456", font=("Roboto", 12))
        self.student_id_label.pack()

        self.class_activity_label = tk.Label(self.export_frame, text="Lớp sinh hoạt: Lớp B", font=("Roboto", 12))
        self.class_activity_label.pack()

        self.absent_number_label = tk.Label(self.export_frame, text="Số vắng: 1000000", font=("Roboto", 12))
        self.absent_number_label.pack()

        self.export_frame.pack_forget()  # Ẩn attendance frame ban đầu
        ################################################################
        #thêm frame vào 1 list frame
        self.frames.append(self.attendance_frame_diemdanh)
        self.frames.append(self.export_frame)
        self.frames.append(self.camera_frame)



    ################################################################################################################################
    ################################
    def update(self):
        if self.vid is not None and self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)
                self.canvas.create_image(0, 0, image=frame, anchor=tk.NW)
                self.canvas.image = frame
            self.root.after(10, self.update)

    def on_closing(self):
        if self.vid is not None:
            self.vid.release()
        self.root.destroy()
    
    def start_camera(self):
        # Mở camera khi hành động Xuất được thực hiện
        if self.vid is None:
            self.vid = cv2.VideoCapture(self.video_source)
            self.update()
    def stop_camera(self):
        if self.vid is not None:
            self.vid.release()
            self.vid = None  #
    #----------------------------------------------------------------
    def toggle_menu(self):
        if self.menu_frame.winfo_ismapped():
            self.menu_frame.pack_forget()
        else:
            self.menu_frame.pack(side=tk.LEFT, fill=tk.Y)

    
    
    def menu_action(self, action):
        # Ẩn tất cả các frame
        for frame in self.frames:
            frame.pack_forget()
            

        if action == "Điểm danh":
            self.attendance_frame_diemdanh.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            self.camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.start_camera()

        elif action == "Xuất":
            self.stop_camera()
            self.export_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
        else:
            messagebox.showinfo("Thông báo", f"Bạn đã chọn: {action}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)  # Khởi tạo lớp AttendanceApp
    root.mainloop()  # 