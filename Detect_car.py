import cv2
import time
from ultralytics import YOLO
from datetime import datetime, date
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import certifi

# Load environment variables from .env file
load_dotenv(dotenv_path='new.env')

# Vehicle class IDs in YOLOv8 (COCO dataset)
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    8: 'van'  # Note: YOLO doesn't specifically have 'van', but we'll use this for similar vehicles
}

class YOLOVehicleCounter:
    def __init__(self, gui_callback, video_source="sample_cars.mp4", counting_zone_width=30):
        self.model = YOLO("yolov8n.pt")
        
        # Use a sample video if available, otherwise default to 0 (webcam)
        if not os.path.exists(video_source):
            print(f"[WARNING] Video file '{video_source}' not found. Using default sample.")
            video_source = 0  # Fallback to webcam if no video file
        
        self.cap = cv2.VideoCapture(video_source)
        
        # Get frame width to calculate center position
        ret, frame = self.cap.read()
        if ret:
            self.line_position = frame.shape[1] // 2  # Center line X position
        else:
            self.line_position = 640  # Default value if frame can't be read
        
        self.counting_zone_width = counting_zone_width
        self.tracks = {}
        self.next_id = 0
        self.gui_callback = gui_callback  # callback to update GUI
        self.vehicle_records = {}  # Records storing check-in/out info per ID
        self.vehicle_counts = {v_type: {'IN': 0, 'OUT': 0} for v_type in VEHICLE_CLASSES.values()}

    def _detect(self, frame):
        results = self.model(frame)[0]
        detections = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if int(class_id) in VEHICLE_CLASSES and score > 0.5:
                vehicle_type = VEHICLE_CLASSES[int(class_id)]
                print(f"[DEBUG] {vehicle_type} detected with confidence {score}")
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    'type': vehicle_type
                })
        return detections

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        if (boxAArea + boxBArea - interArea) == 0:
            return 0
        return interArea / float(boxAArea + boxBArea - interArea)

    def _update_tracks(self, detections):
        updated_tracks = {}
        assigned_detections = set()

        for track_id, track in self.tracks.items():
            best_iou = 0
            best_det_idx = -1
            for i, det in enumerate(detections):
                if i in assigned_detections:
                    continue
                iou = self._calculate_iou(track['bbox'], det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i

            if best_iou > 0.3:
                updated_tracks[track_id] = {
                    'bbox': detections[best_det_idx]['bbox'],
                    'type': detections[best_det_idx]['type'],
                    'last_seen': time.time(),
                    'crossed': track['crossed'],
                    'initial_x': track['initial_x']
                }
                assigned_detections.add(best_det_idx)

        # Add new detections as new tracks
        for i, det in enumerate(detections):
            if i not in assigned_detections:
                updated_tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'type': det['type'],
                    'last_seen': time.time(),
                    'crossed': False,
                    'initial_x': det['bbox'][0]
                }
                self.next_id += 1

        # Remove tracks unseen for more than 3 seconds
        self.tracks = {tid: t for tid, t in updated_tracks.items() if time.time() - t['last_seen'] < 3.0}

    def _check_crossing(self):
        for track_id, track in self.tracks.items():
            print(f"[DEBUG] Checking ID {track_id}, crossed={track['crossed']}")
            if track['crossed']:
                continue
            current_x = track['bbox'][0]
            initial_x = track['initial_x']
            print(f"[DEBUG] Track {track_id} moved from {initial_x} to {current_x}")
            time_str = datetime.now().strftime("%H:%M:%S")
            date_str = datetime.now().strftime("%Y-%m-%d")

            # Crossing left to right (check-in)
            if initial_x < self.line_position and current_x > self.line_position + self.counting_zone_width:
                print(f"[DEBUG] Vehicle {track_id} crossed IN")
                track['crossed'] = True
                self.vehicle_counts[track['type']]['IN'] += 1
                self.vehicle_records[track_id] = {
                    "date": date_str,
                    "check_in_time": time_str,
                    "check_out_time": None,
                    "direction": "IN",
                    "vehicle_type": track['type']
                }
                self.gui_callback(track_id, "IN", time_str, track['type'])

            # Crossing right to left (check-out)
            elif initial_x > self.line_position and current_x < self.line_position - self.counting_zone_width:
                print(f"[DEBUG] Vehicle {track_id} crossed OUT")
                track['crossed'] = True
                self.vehicle_counts[track['type']]['OUT'] += 1
                if track_id in self.vehicle_records:
                    self.vehicle_records[track_id]["check_out_time"] = time_str
                    self.vehicle_records[track_id]["direction"] = "OUT"
                else:
                    self.vehicle_records[track_id] = {
                        "date": date_str,
                        "check_in_time": None,
                        "check_out_time": time_str,
                        "direction": "OUT",
                        "vehicle_type": track['type']
                    }
                self.gui_callback(track_id, "OUT", time_str, track['type'])

    def _update_display(self, frame):
        h, w = frame.shape[:2]
        
        # Recalculate center line position based on current frame
        self.line_position = w // 2

        # Draw center red line
        cv2.line(frame, (self.line_position, 0), (self.line_position, h), (0, 0, 255), 3)

        # Draw counting zone boundary lines (orange)
        cv2.line(frame, (self.line_position - self.counting_zone_width, 0),
                (self.line_position - self.counting_zone_width, h), (0, 165, 255), 2)
        cv2.line(frame, (self.line_position + self.counting_zone_width, 0),
                (self.line_position + self.counting_zone_width, h), (0, 165, 255), 2)

        for track_id, track in self.tracks.items():
            x, y, w_box, h_box = track['bbox']
            # Different colors for different vehicle types
            if track['type'] == 'car':
                color = (255, 0, 0)  # Blue
            elif track['type'] == 'truck':
                color = (0, 255, 0)  # Green
            elif track['type'] == 'bus':
                color = (0, 0, 255)  # Red
            elif track['type'] == 'motorcycle':
                color = (255, 255, 0)  # Cyan
            else:  # van or others
                color = (0, 255, 255)  # Yellow
            
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(frame, f"{track['type']} {track_id}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show total counts for each vehicle type
        y_offset = 30
        for v_type, counts in self.vehicle_counts.items():
            cv2.putText(frame, f"{v_type}: IN {counts['IN']} OUT {counts['OUT']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                # Loop the video if it ends
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            detections = self._detect(frame)
            self._update_tracks(detections)
            self._check_crossing()
            self._update_display(frame)

            cv2.imshow("Vehicle Counter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


class VehicleCounterGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLO Vehicle Counter Log - MongoDB")
        
        # MongoDB Connection Frame
        conn_frame = tk.Frame(self.root)
        conn_frame.pack(pady=10)
        
        tk.Label(conn_frame, text="MongoDB URI:").grid(row=0, column=0, padx=5)
        self.uri_entry = tk.Entry(conn_frame, width=50)
        self.uri_entry.grid(row=0, column=1, padx=5)
        self.uri_entry.insert(0, os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
        
        tk.Label(conn_frame, text="Database:").grid(row=1, column=0, padx=5)
        self.db_entry = tk.Entry(conn_frame)
        self.db_entry.grid(row=1, column=1, padx=5, sticky="ew")
        self.db_entry.insert(0, "vehicle_counter")
        
        tk.Label(conn_frame, text="Collection:").grid(row=2, column=0, padx=5)
        self.collection_entry = tk.Entry(conn_frame)
        self.collection_entry.grid(row=2, column=1, padx=5, sticky="ew")
        self.collection_entry.insert(0, "movements")
        
        self.connect_button = tk.Button(conn_frame, text="Connect", command=self.connect_to_mongodb)
        self.connect_button.grid(row=3, column=1, pady=5, sticky="e")
        
        # Status label
        self.status_label = tk.Label(conn_frame, text="Not connected", fg="red")
        self.status_label.grid(row=3, column=0, sticky="w")
        
        # Treeview for displaying data
        self.tree = ttk.Treeview(self.root, columns=("ID", "Type", "Date", "Direction", "Check In", "Check Out"), show='headings')
        self.tree.heading("ID", text="ID")
        self.tree.heading("Type", text="Vehicle Type")
        self.tree.heading("Date", text="Date")
        self.tree.heading("Direction", text="Direction")
        self.tree.heading("Check In", text="Check In Time")
        self.tree.heading("Check Out", text="Check Out Time")
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Buttons frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        self.save_button = tk.Button(btn_frame, text="Save to MongoDB", command=self.save_to_mongodb, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        self.load_button = tk.Button(btn_frame, text="Load from MongoDB", command=self.load_from_mongodb, state=tk.DISABLED)
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        # Initialize MongoDB client
        self.client = None
        self.db = None
        self.collection = None
        
        self.counter = YOLOVehicleCounter(self.add_or_update_entry)
        self.thread = threading.Thread(target=self.counter.run, daemon=True)
        self.thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(1000, self.refresh_table)
        self.root.mainloop()
    
    def connect_to_mongodb(self):
        uri = self.uri_entry.get()
        db_name = self.db_entry.get()
        collection_name = self.collection_entry.get()

        try:
            if uri.startswith("mongodb+srv://"):
                self.client = MongoClient(uri, tlsCAFile=certifi.where())
            else:
                self.client = MongoClient(uri)

            self.client.admin.command('ping')

            self.db = self.client[db_name]
            self.collection = self.db[collection_name]

            self.status_label.config(text="Connected", fg="green")
            self.save_button.config(state=tk.NORMAL)
            self.load_button.config(state=tk.NORMAL)
            messagebox.showinfo("Success", "Connected to MongoDB successfully!")

        except Exception as e:
            self.status_label.config(text="Connection failed", fg="red")
            print("MongoDB Connection Error:", e)
            messagebox.showerror("Connection Error", f"Failed to connect to MongoDB: {str(e)}")

    
    def add_or_update_entry(self, vid, direction, time_str, vehicle_type):
        existing = None
        for child in self.tree.get_children():
            vals = self.tree.item(child)["values"]
            if vals[0] == vid:
                existing = child
                break

        date_str = datetime.now().strftime("%Y-%m-%d")
        
        if existing:
            vals = list(self.tree.item(existing)["values"])
            if direction == "IN":
                vals[1] = vehicle_type
                vals[2] = date_str
                vals[3] = direction
                vals[4] = time_str
            elif direction == "OUT":
                vals[1] = vehicle_type
                vals[2] = date_str
                vals[3] = direction
                vals[5] = time_str
            self.tree.item(existing, values=vals)
        else:
            if direction == "IN":
                self.tree.insert("", "end", values=(vid, vehicle_type, date_str, direction, time_str, ""))
            else:
                self.tree.insert("", "end", values=(vid, vehicle_type, date_str, direction, "", time_str))

    def refresh_table(self):
        for vid, rec in self.counter.vehicle_records.items():
            existing = None
            for child in self.tree.get_children():
                vals = self.tree.item(child)["values"]
                if vals[0] == vid:
                    existing = child
                    break
            if existing:
                vals = list(self.tree.item(existing)["values"])
                vals[1] = rec.get("vehicle_type", "")
                vals[2] = rec.get("date", "")
                vals[3] = rec.get("direction", "")
                vals[4] = rec.get("check_in_time", "") or ""
                vals[5] = rec.get("check_out_time", "") or ""
                self.tree.item(existing, values=vals)
            else:
                self.tree.insert("", "end", values=(
                    vid,
                    rec.get("vehicle_type", ""),
                    rec.get("date", ""),
                    rec.get("direction", ""),
                    rec.get("check_in_time", "") or "",
                    rec.get("check_out_time", "") or ""
                ))

        self.root.after(1000, self.refresh_table)

    def save_to_mongodb(self):
        if not self.collection:
            messagebox.showerror("Error", "Not connected to MongoDB")
            return
        
        try:
            # Convert records to list of documents
            documents = []
            for vid, rec in self.counter.vehicle_records.items():
                doc = {
                    "vehicle_id": vid,
                    "vehicle_type": rec.get("vehicle_type"),
                    "date": rec.get("date"),
                    "direction": rec.get("direction"),
                    "check_in_time": rec.get("check_in_time"),
                    "check_out_time": rec.get("check_out_time"),
                    "timestamp": datetime.now()
                }
                documents.append(doc)
            
            # Insert documents
            result = self.collection.insert_many(documents)
            
            # Clear local records after saving
            self.counter.vehicle_records = {}
            for item in self.tree.get_children():
                self.tree.delete(item)
                
            messagebox.showinfo("Success", f"Saved {len(result.inserted_ids)} records to MongoDB")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save to MongoDB: {str(e)}")

    def load_from_mongodb(self):
        if not self.collection:
            messagebox.showerror("Error", "Not connected to MongoDB")
            return
        
        try:
            # Clear current treeview
            for item in self.tree.get_children():
                self.tree.delete(item)
                
            # Query all documents, sorted by timestamp
            documents = self.collection.find().sort("timestamp", -1)
            
            # Limit to 100 most recent records for display
            count = 0
            for doc in documents:
                if count >= 100:
                    break
                
                self.tree.insert("", "end", values=(
                    doc.get("vehicle_id"),
                    doc.get("vehicle_type"),
                    doc.get("date"),
                    doc.get("direction"),
                    doc.get("check_in_time", ""),
                    doc.get("check_out_time", "")
                ))
                count += 1
                
            messagebox.showinfo("Success", f"Loaded {count} records from MongoDB")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load from MongoDB: {str(e)}")

    def on_closing(self):
        if self.client:
            self.client.close()
        self.counter.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    VehicleCounterGUI()