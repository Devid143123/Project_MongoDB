# Detect_people_SQLite.py

import cv2
import time
from ultralytics import YOLO
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import sqlite3
from tkcalendar import DateEntry

class YOLOPersonCounter:
    def __init__(self, gui_callback, video_source=None, counting_zone_width=30):
        self.model = YOLO("yolov8n.pt")
        sample_video_path = "people.mp4"
        self.cap = cv2.VideoCapture(sample_video_path if os.path.exists(sample_video_path) else 0)
        ret, frame = self.cap.read()
        self.line_position = frame.shape[1] // 2 if ret else 360
        self.counting_zone_width = counting_zone_width
        self.tracks = {}  # person_id: {'bbox': ..., 'last_seen': ..., 'crossed': ..., 'initial_x': ..., 'last_count_time': ..., 'frames_tracked': ...}
        self.next_id = 0
        self.gui_callback = gui_callback
        self.person_records = {}
        self.counts = {"IN": 0, "OUT": 0}

    def _detect(self, frame):
        results = self.model(frame)[0]
        detections = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if int(class_id) == 0 and score > 0.5:  # 0 is class ID for person
                detections.append({'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1))})
        return detections

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) != 0 else 0

    def _update_tracks(self, detections):
        updated_tracks = {}
        assigned_detections = set()
        for track_id, track in self.tracks.items():
            best_iou, best_det_idx = 0, -1
            for i, det in enumerate(detections):
                if i in assigned_detections:
                    continue
                iou = self._calculate_iou(track['bbox'], det['bbox'])
                if iou > best_iou:
                    best_iou, best_det_idx = iou, i
            if best_iou > 0.3:
                updated_tracks[track_id] = {
                    'bbox': detections[best_det_idx]['bbox'],
                    'last_seen': time.time(),
                    'crossed': track['crossed'], 'last_count_time': track.get('last_count_time', 0), 'frames_tracked': track.get('frames_tracked', 0) + 1,
                    'initial_x': track['initial_x']
                }
                assigned_detections.add(best_det_idx)
        for i, det in enumerate(detections):
            if i not in assigned_detections:
                updated_tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'last_seen': time.time(),
                    'crossed': False,
                    'initial_x': det['bbox'][0], 'last_count_time': time.time(), 'frames_tracked': 1
                }
                self.next_id += 1
        self.tracks = {tid: t for tid, t in updated_tracks.items() if time.time() - t['last_seen'] < 3.0}

    def _check_crossing(self):
        for track_id, track in self.tracks.items():
            if track['crossed'] or track.get('frames_tracked', 0) < 5:
                continue
            if time.time() - track.get('last_count_time', 0) < 1.0:
                continue
            current_x, initial_x = track['bbox'][0], track['initial_x']
            time_str = datetime.now().strftime("%H:%M:%S")
            date_str = datetime.now().strftime("%Y-%m-%d")
            if initial_x < self.line_position and current_x > self.line_position + self.counting_zone_width:
                track['crossed'] = True
                track['last_count_time'] = time.time()
                self.counts['IN'] += 1
                self.person_records[track_id] = {"date": date_str, "time": time_str, "direction": "IN"}
                self.gui_callback(track_id, "IN", time_str)
            elif initial_x > self.line_position and current_x < self.line_position - self.counting_zone_width:
                track['crossed'] = True
                track['last_count_time'] = time.time()
                self.counts['OUT'] += 1
                self.person_records[track_id] = {"date": date_str, "time": time_str, "direction": "OUT"}
                self.gui_callback(track_id, "OUT", time_str)

    def _update_display(self, frame):
        self.line_position = int(frame.shape[1] * 0.7)
        cv2.line(frame, (self.line_position, 0), (self.line_position, frame.shape[0]), (0, 0, 255), 4)
        for track_id, track in self.tracks.items():
            x, y, w_box, h_box = track['bbox']
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset = 30
        for direction, count in self.counts.items():
            label = "Check In" if direction == "IN" else "Check Out"
            cv2.putText(frame, f"{label}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            detections = self._detect(frame)
            self._update_tracks(detections)
            self._check_crossing()
            self._update_display(frame)
            cv2.imshow("Person Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

class PersonCounterGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLO Person Counter Log - SQLite")
        self.init_sqlite()
        
        # Create filter frame
        filter_frame = tk.Frame(self.root)
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Date range filter
        tk.Label(filter_frame, text="From:").pack(side=tk.LEFT)
        self.from_date = DateEntry(filter_frame)
        self.from_date.pack(side=tk.LEFT, padx=5)
        
        tk.Label(filter_frame, text="To:").pack(side=tk.LEFT)
        self.to_date = DateEntry(filter_frame)
        self.to_date.pack(side=tk.LEFT, padx=5)
        
        # Direction filter
        self.direction_var = tk.StringVar(value="ALL")
        tk.Label(filter_frame, text="Direction:").pack(side=tk.LEFT, padx=(10,0))
        tk.Radiobutton(filter_frame, text="All", variable=self.direction_var, value="ALL").pack(side=tk.LEFT)
        tk.Radiobutton(filter_frame, text="Check In", variable=self.direction_var, value="IN").pack(side=tk.LEFT)
        tk.Radiobutton(filter_frame, text="Check Out", variable=self.direction_var, value="OUT").pack(side=tk.LEFT)
        
        # Filter buttons
        tk.Button(filter_frame, text="Filter", command=self.filter_records).pack(side=tk.LEFT, padx=5)
        tk.Button(filter_frame, text="Reset", command=self.reset_filter).pack(side=tk.LEFT)
        
        # Create treeview
        self.tree = ttk.Treeview(self.root, columns=("ID", "Date", "Time", "Check"), show='headings')
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor='center')
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(button_frame, text="Save to SQLite", command=self.save_to_sqlite).pack(side=tk.LEFT)
        tk.Button(button_frame, text="Refresh", command=self.load_records).pack(side=tk.LEFT, padx=5)
        
        self.counter = YOLOPersonCounter(self.add_entry)
        threading.Thread(target=self.counter.run, daemon=True).start()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Load initial records
        self.load_records()
        self.root.mainloop()

    def init_sqlite(self):
        db_path = os.path.expanduser("~/Documents/people_counter.db")
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS person_logs (
                person_id INTEGER,
                date TEXT,
                time TEXT,
                direction TEXT,
                timestamp TEXT
            )
        """)
        self.conn.commit()

    def load_records(self):
        """Load all records from database"""
        self.tree.delete(*self.tree.get_children())
        self.cursor.execute("SELECT person_id, date, time, direction FROM person_logs ORDER BY date DESC, time DESC")
        for row in self.cursor.fetchall():
            self.tree.insert("", "end", values=(row[0], row[1], row[2], "Check In" if row[3] == "IN" else "Check Out"))

    def filter_records(self):
        """Filter records by date range and direction"""
        from_date = self.from_date.get_date().strftime("%Y-%m-%d")
        to_date = self.to_date.get_date().strftime("%Y-%m-%d")
        direction = self.direction_var.get()
        
        query = """
            SELECT person_id, date, time, direction FROM person_logs 
            WHERE date BETWEEN ? AND ?
        """
        params = [from_date, to_date]
        
        if direction != "ALL":
            query += " AND direction = ?"
            params.append(direction)
            
        query += " ORDER BY date DESC, time DESC"
        
        self.tree.delete(*self.tree.get_children())
        self.cursor.execute(query, tuple(params))
        
        for row in self.cursor.fetchall():
            self.tree.insert("", "end", values=(row[0], row[1], row[2], "Check In" if row[3] == "IN" else "Check Out"))

    def reset_filter(self):
        """Reset all filters and show all records"""
        self.direction_var.set("ALL")
        self.load_records()

    def save_to_sqlite(self):
        try:
            for pid, rec in self.counter.person_records.items():
                self.cursor.execute("""
                    INSERT INTO person_logs (person_id, date, time, direction, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (pid, rec['date'], rec['time'], rec['direction'], datetime.now().isoformat()))
            self.conn.commit()
            messagebox.showinfo("Success", "Records saved to SQLite successfully.")
            self.counter.person_records.clear()
            self.load_records()  # Refresh the treeview with updated records
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def add_entry(self, pid, direction, time_str):
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.tree.insert("", "end", values=(pid, date_str, time_str, "Check In" if direction == "IN" else "Check Out"))

    def on_closing(self):
        self.conn.close()
        self.counter.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    PersonCounterGUI()