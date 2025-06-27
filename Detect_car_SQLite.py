import cv2
import time
from ultralytics import YOLO
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.ndimage import gaussian_filter
# Vehicle class IDs in YOLOv8 (COCO dataset)
VEHICLE_CLASSES = {
    2: 'Car',          # car
    3: 'Bike',         # motorcycle, bike, etc.
    5: 'Oil Truck',    # bus (we'll treat buses as Oil Trucks)
    7: 'Oil Truck',    # truck
    8: 'Oil Truck'     # van (we'll treat vans as Oil Trucks)
}
# Color mappings for each vehicle type (BGR format)
VEHICLE_COLORS = {
    'Oil Truck': (0, 165, 255),  # Orange (BGR)
    'Car': (255, 0, 0),          # Blue
    'Bike': (0, 255, 0)          # Green
}
class YOLOVehicleCounter:
    def __init__(self, gui_callback, video_source=None, counting_zone_height=30):
        self.model = YOLO("yolov8n.pt")
        sample_video_path = "sample_cars.mp4"
        self.cap = cv2.VideoCapture(sample_video_path if os.path.exists(sample_video_path) else 0)
        ret, frame = self.cap.read()
        self.line_position = frame.shape[0] // 2 if ret else 360
        self.counting_zone_height = counting_zone_height
        self.tracks = {}
        self.next_id = 0
        self.gui_callback = gui_callback
        self.vehicle_records = {}
        self.vehicle_counts = {v_type: {'IN': 0, 'OUT': 0} for v_type in VEHICLE_CLASSES.values()}

    def _detect(self, frame):
        results = self.model(frame)[0]
        detections = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if int(class_id) in VEHICLE_CLASSES and score > 0.5:
                vehicle_type = VEHICLE_CLASSES[int(class_id)]
                detections.append({'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)), 'type': vehicle_type})
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
                    'type': detections[best_det_idx]['type'],
                    'last_seen': time.time(),
                    'crossed': track['crossed'],
                    'initial_y': track['initial_y']
                }
                assigned_detections.add(best_det_idx)
        for i, det in enumerate(detections):
            if i not in assigned_detections:
                updated_tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'type': det['type'],
                    'last_seen': time.time(),
                    'crossed': False,
                    'initial_y': det['bbox'][1]
                }
                self.next_id += 1
        self.tracks = {tid: t for tid, t in updated_tracks.items() if time.time() - t['last_seen'] < 3.0}

    def _check_crossing(self):
        for track_id, track in self.tracks.items():
            if track['crossed']:
                continue
            current_y, initial_y = track['bbox'][1], track['initial_y']
            time_str = datetime.now().strftime("%H:%M:%S")
            date_str = datetime.now().strftime("%Y-%m-%d")
            if initial_y < self.line_position and current_y > self.line_position + self.counting_zone_height:
                track['crossed'] = True
                self.vehicle_counts[track['type']]['IN'] += 1
                self.vehicle_records[track_id] = {"date": date_str, "check_in_time": time_str, "check_out_time": None, "direction": "IN", "vehicle_type": track['type']}
                self.gui_callback(track_id, "IN", time_str, track['type'])
            elif initial_y > self.line_position and current_y < self.line_position - self.counting_zone_height:
                track['crossed'] = True
                self.vehicle_counts[track['type']]['OUT'] += 1
                if track_id in self.vehicle_records:
                    self.vehicle_records[track_id]["check_out_time"] = time_str
                    self.vehicle_records[track_id]["direction"] = "OUT"
                else:
                    self.vehicle_records[track_id] = {"date": date_str, "check_in_time": None, "check_out_time": time_str, "direction": "OUT", "vehicle_type": track['type']}
                self.gui_callback(track_id, "OUT", time_str, track['type'])
    
    def _update_display(self, frame):
        self.line_position = int(frame.shape[0] * 0.70)
        cv2.line(frame, (0, self.line_position), (frame.shape[1], self.line_position), (0, 0, 255), 4)
        for track_id, track in self.tracks.items():
            x, y, w_box, h_box = track['bbox']
            # Use custom colors based on vehicle type
            color = VEHICLE_COLORS.get(track['type'], (0, 255, 255))  # Default: Yellow if unknown
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(frame, f"{track['type']} {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset = 30
        for v_type in ['Oil Truck', 'Car', 'Bike']:  # Display only these 3 types
            counts = self.vehicle_counts[v_type]
            color = VEHICLE_COLORS[v_type]
            cv2.putText(frame, f"{v_type}: IN {counts['IN']} OUT {counts['OUT']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
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
            cv2.imshow("Vehicle Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

class VehicleCounterGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLO Vehicle Counter Log - SQLite")
        self.init_sqlite()
        self.tree = ttk.Treeview(self.root, columns=("ID", "Type", "Date", "Direction", "Check In", "Check Out"), show='headings')
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor='center')
        self.tree.pack(fill=tk.BOTH, expand=True)
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        # Filter frame
        filter_frame = tk.Frame(self.root)
        filter_frame.pack(pady=5)

        tk.Label(filter_frame, text="From Date (YYYY-MM-DD):").pack(side=tk.LEFT, padx=5)
        self.date_from_var = tk.StringVar()
        self.date_from_entry = tk.Entry(filter_frame, textvariable=self.date_from_var, width=12)
        self.date_from_entry.pack(side=tk.LEFT)

        tk.Label(filter_frame, text="To Date (YYYY-MM-DD):").pack(side=tk.LEFT, padx=5)
        self.date_to_var = tk.StringVar()
        self.date_to_entry = tk.Entry(filter_frame, textvariable=self.date_to_var, width=12)
        self.date_to_entry.pack(side=tk.LEFT)

        tk.Label(filter_frame, text="Filter by Type:").pack(side=tk.LEFT, padx=5)
        self.vehicle_type_var = tk.StringVar(value="All")
        # Only show 3 categories now
        type_options = ["All", "Oil Truck", "Car", "Bike"]
        self.type_dropdown = ttk.Combobox(filter_frame, textvariable=self.vehicle_type_var, values=type_options, state="readonly", width=12)
        self.type_dropdown.pack(side=tk.LEFT)

        tk.Label(filter_frame, text="Filter by Direction:").pack(side=tk.LEFT, padx=5)
        self.direction_var = tk.StringVar(value="All")
        dir_options = ["All", "IN", "OUT"]
        self.direction_dropdown = ttk.Combobox(filter_frame, textvariable=self.direction_var, values=dir_options, state="readonly", width=10)
        self.direction_dropdown.pack(side=tk.LEFT)

        tk.Button(filter_frame, text="Apply Filters", command=self.load_from_sqlite).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Save to SQLite", command=self.save_to_sqlite).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Load from SQLite", command=self.load_from_sqlite).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Export to Excel", command=self.export_to_excel).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Plot Chart", command=self.plot_chart).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Export Chart", command=self.export_chart_to_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Export PDF Report", command=self.export_pdf_report).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Print All Records", command=self.print_all_records).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Clear All Records", command=self.clear_all_records).pack(side=tk.LEFT, padx=5)
        self.counter = YOLOVehicleCounter(self.add_or_update_entry)
        threading.Thread(target=self.counter.run, daemon=True).start()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(1000, self.refresh_table)
        self.load_from_sqlite()  # Load most recent records when GUI starts
        self.root.mainloop()

    def init_sqlite(self):
        db_path = os.path.expanduser("~/Documents/total_energies_vehicle_counter.db")
        print(">>> SQLite DB path:", db_path)
        self.sqlite_conn = sqlite3.connect(db_path)
        self.sqlite_cursor = self.sqlite_conn.cursor()
        self.sqlite_cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_logs (
                vehicle_id INTEGER,
                vehicle_type TEXT,
                date TEXT,
                direction TEXT,
                check_in_time TEXT,
                check_out_time TEXT,
                timestamp TEXT
            )
        """)
        self.sqlite_conn.commit()

    def save_to_sqlite(self):
        try:
            for vid, rec in self.counter.vehicle_records.items():
                self.sqlite_cursor.execute("""
                    INSERT INTO vehicle_logs (vehicle_id, vehicle_type, date, direction, check_in_time, check_out_time, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    vid,
                    rec.get("vehicle_type"),
                    rec.get("date"),
                    rec.get("direction"),
                    rec.get("check_in_time"),
                    rec.get("check_out_time"),
                    datetime.now().isoformat()
                ))
            self.sqlite_conn.commit()
            messagebox.showinfo("Success", "Records saved to SQLite successfully.")
            self.counter.vehicle_records.clear()
            for item in self.tree.get_children():
                self.tree.delete(item)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save to SQLite: {str(e)}")

    def load_from_sqlite(self):
        try:
            self.tree.delete(*self.tree.get_children())
            query = "SELECT vehicle_id, vehicle_type, date, direction, check_in_time, check_out_time FROM vehicle_logs"
            filters = []
            params = []

            selected_type = self.vehicle_type_var.get()
            selected_dir = self.direction_var.get()

            if selected_type != "All":
                filters.append("vehicle_type = ?")
                params.append(selected_type)

            if selected_dir != "All":
                filters.append("direction = ?")
                params.append(selected_dir)

            if self.date_from_var.get():
                filters.append("date >= ?")
                params.append(self.date_from_var.get())

            if self.date_to_var.get():
                filters.append("date <= ?")
                params.append(self.date_to_var.get())

            if filters:
                query += " WHERE " + " AND ".join(filters)

            query += " ORDER BY timestamp DESC LIMIT 100"
            self.sqlite_cursor.execute(query, params)

            for row in self.sqlite_cursor.fetchall():
                self.tree.insert("", "end", values=row)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load from SQLite: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load from SQLite: {str(e)}")

    def export_to_excel(self):
        try:
            data = [self.tree.item(child)["values"] for child in self.tree.get_children()]
            if not data:
                messagebox.showinfo("No Data", "No records to export.")
                return
            df = pd.DataFrame(data, columns=["ID", "Type", "Date", "Direction", "Check In", "Check Out"])
            filename = f"vehicle_counter_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            df.to_excel(filename, index=False)
            messagebox.showinfo("Success", f"Filtered data exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export to Excel: {str(e)}")

    def export_pdf_report(self):
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            import tempfile

            # Prepare chart
            vehicle_types = list(self.counter.vehicle_counts.keys())
            in_counts = [self.counter.vehicle_counts[vt]['IN'] for vt in vehicle_types]
            out_counts = [self.counter.vehicle_counts[vt]['OUT'] for vt in vehicle_types]
            fig, ax = plt.subplots(figsize=(8, 4))
            x = range(len(vehicle_types))
            width = 0.35
            ax.bar([i - width/2 for i in x], in_counts, width, label='IN', color='green')
            ax.bar([i + width/2 for i in x], out_counts, width, label='OUT', color='red')
            ax.set_xlabel('Vehicle Type')
            ax.set_ylabel('Count')
            ax.set_title('Vehicle Counts by Type and Direction')
            ax.set_xticks(x)
            ax.set_xticklabels(vehicle_types)
            ax.legend()
            fig.tight_layout()

            # Save data to table
            data = [self.tree.item(child)["values"] for child in self.tree.get_children()]
            df = pd.DataFrame(data, columns=["ID", "Type", "Date", "Direction", "Check In", "Check Out"])

            # Add duration column
            def calc_duration(row):
                if row['Check In'] and row['Check Out']:
                    in_time = datetime.strptime(row['Check In'], "%H:%M:%S")
                    out_time = datetime.strptime(row['Check Out'], "%H:%M:%S")
                    delta = (out_time - in_time).total_seconds() / 60.0
                    return round(delta, 2)
                return None
            df["Duration (min)"] = df.apply(calc_duration, axis=1)

            # Save to PDF
            filename = f"vehicle_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            with PdfPages(filename) as pdf:
                page_num = 1
                # Add Company Title Page
                fig_cover, ax_cover = plt.subplots(figsize=(8.5, 11))
                ax_cover.axis('off')
                ax_cover.text(0.5, 0.8, "Total Energies", fontsize=24, weight='bold', ha='center')
                ax_cover.text(0.5, 0.7, "Vehicle Detection Report", fontsize=18, ha='center')
                ax_cover.text(0.5, 0.6, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=12, ha='center')
                ax_cover.text(0.5, 0.5, f"Total Records: {len(df)}", fontsize=12, ha='center')
                total_in = df[df['Direction'] == 'IN'].shape[0]
                total_out = df[df['Direction'] == 'OUT'].shape[0]
                avg_duration = df['Duration (min)'].dropna().mean()
                ax_cover.text(0.5, 0.45, f"Total IN: {total_in} | Total OUT: {total_out}", fontsize=12, ha='center')
                ax_cover.text(0.5, 0.4, f"Average Duration: {round(avg_duration, 2) if not pd.isna(avg_duration) else 'N/A'} min", fontsize=12, ha='center')
                ax_cover.text(0.5, 0.1, f"Page {page_num} | Generated by Total Energies System", fontsize=10, ha='center', color='gray')
                pdf.savefig(fig_cover)
                plt.close(fig_cover)
                page_num += 1

                # Add Summary Table Page
                summary_data = [["Metric", "Value"],
                                ["Total Records", str(len(df))],
                                ["Total IN", str(total_in)],
                                ["Total OUT", str(total_out)],
                                ["Average Duration", f"{round(avg_duration, 2)} min" if not pd.isna(avg_duration) else "N/A"]]
                summary_fig, summary_ax = plt.subplots(figsize=(6, 1.5))
                summary_ax.axis('off')
                summary_table = summary_ax.table(cellText=summary_data, cellLoc='center', loc='center')
                summary_table.scale(1, 1.5)
                summary_table.auto_set_font_size(False)
                summary_table.set_fontsize(10)
                summary_ax.text(0.5, -0.3, f"Page {page_num} | Generated by Total Energies System", fontsize=10, ha='center', color='gray')
                pdf.savefig(summary_fig)
                plt.close(summary_fig)
                page_num += 1

                # Add chart page
                fig.subplots_adjust(bottom=0.15)
                fig.text(0.5, 0.02, f"Page {page_num} | Generated by Total Energies System", ha='center', fontsize=10, color='gray')
                pdf.savefig(fig)
                plt.close(fig)
                page_num += 1

                # Add table page
                fig2, ax2 = plt.subplots(figsize=(8.5, 11))
                ax2.axis('off')
                ax2.set_title('Vehicle Logs Table', fontsize=14, weight='bold')
                ax2.axis('tight')
                table = ax2.table(cellText=df.values,
                                  colLabels=df.columns,
                                  cellLoc='center',
                                  loc='center')
                table.scale(1, 1.4)
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.auto_set_column_width(col=list(range(len(df.columns))))
                ax2.text(0.5, 0.02, f"Page {page_num} | Generated by Total Energies System", fontsize=10, ha='center', color='gray')
                pdf.savefig(fig2)
                plt.close(fig2)

            messagebox.showinfo("Success", f"PDF report exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export PDF: {str(e)}")

    def export_chart_to_image(self):
        try:
            vehicle_types = list(self.counter.vehicle_counts.keys())
            in_counts = [self.counter.vehicle_counts[vt]['IN'] for vt in vehicle_types]
            out_counts = [self.counter.vehicle_counts[vt]['OUT'] for vt in vehicle_types]
            fig, ax = plt.subplots(figsize=(8, 5))
            x = range(len(vehicle_types))
            width = 0.35
            ax.bar([i - width/2 for i in x], in_counts, width, label='IN', color='green')
            ax.bar([i + width/2 for i in x], out_counts, width, label='OUT', color='red')
            ax.set_xlabel('Vehicle Type')
            ax.set_ylabel('Count')
            ax.set_title('Vehicle Counts by Type and Direction')
            ax.set_xticks(x)
            ax.set_xticklabels(vehicle_types)
            ax.legend()
            fig.tight_layout()
            filename = f"vehicle_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.savefig(filename)
            messagebox.showinfo("Success", f"Chart exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export chart: {str(e)}")

    # Original export_to_excel replaced
        try:
            data = [self.tree.item(child)["values"] for child in self.tree.get_children()]
            df = pd.DataFrame(data, columns=["ID", "Type", "Date", "Direction", "Check In", "Check Out"])
            filename = f"vehicle_counter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            df.to_excel(filename, index=False)
            messagebox.showinfo("Success", f"Data exported to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export to Excel: {str(e)}")

    def plot_chart(self):
        try:
            vehicle_types = list(self.counter.vehicle_counts.keys())
            in_counts = [self.counter.vehicle_counts[vt]['IN'] for vt in vehicle_types]
            out_counts = [self.counter.vehicle_counts[vt]['OUT'] for vt in vehicle_types]
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Vehicle Count Chart")
            fig, ax = plt.subplots(figsize=(8, 5))
            x = range(len(vehicle_types))
            width = 0.35
            ax.bar([i - width/2 for i in x], in_counts, width, label='IN', color='green')
            ax.bar([i + width/2 for i in x], out_counts, width, label='OUT', color='red')
            ax.set_xlabel('Vehicle Type')
            ax.set_ylabel('Count')
            ax.set_title('Vehicle Counts by Type and Direction')
            ax.set_xticks(x)
            ax.set_xticklabels(vehicle_types)
            ax.legend()
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create plot: {str(e)}")

    def add_or_update_entry(self, vid, direction, time_str, vehicle_type):
        date_str = datetime.now().strftime("%Y-%m-%d")
        existing = None
        for child in self.tree.get_children():
            vals = self.tree.item(child)["values"]
            if vals[0] == vid:
                existing = child
                break
        if existing:
            vals = list(self.tree.item(existing)["values"])
            if direction == "IN":
                vals[1], vals[2], vals[3], vals[4] = vehicle_type, date_str, direction, time_str
            elif direction == "OUT":
                vals[1], vals[2], vals[3], vals[5] = vehicle_type, date_str, direction, time_str
            self.tree.item(existing, values=vals)
        else:
            if direction == "IN":
                self.tree.insert("", "end", values=(vid, vehicle_type, date_str, direction, time_str, ""))
            else:
                self.tree.insert("", "end", values=(vid, vehicle_type, date_str, direction, "", time_str))

    def refresh_table(self):
        for vid, rec in self.counter.vehicle_records.copy().items():
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

    def print_all_records(self):
        try:
            print(">>> All Records from SQLite:")
            self.sqlite_cursor.execute("SELECT * FROM vehicle_logs ORDER BY timestamp DESC")
            rows = self.sqlite_cursor.fetchall()
            for row in rows:
                print(row)
        except Exception as e:
            print("Failed to read from SQLite:", str(e))

    def clear_all_records(self):
        try:
            confirm = messagebox.askyesno("Confirm", "Are you sure you want to delete all records?")
            if confirm:
                self.sqlite_cursor.execute("DELETE FROM vehicle_logs")
                self.sqlite_conn.commit()
                self.tree.delete(*self.tree.get_children())
                messagebox.showinfo("Success", "All records deleted.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete records: {str(e)}")

    def on_closing(self):
        self.sqlite_conn.close()
        self.counter.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    VehicleCounterGUI()
