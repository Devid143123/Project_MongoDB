import cv2
import time
from ultralytics import YOLO
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Vehicle class IDs in YOLOv8 (COCO dataset)
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    8: 'van'
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
        self.last_report_date = datetime.now().date()

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
        self.line_position = int(frame.shape[0] * 0.90)
        cv2.line(frame, (0, self.line_position), (frame.shape[1], self.line_position), (0, 0, 255), 4)
        for track_id, track in self.tracks.items():
            x, y, w_box, h_box = track['bbox']
            color = (255, 0, 0) if track['type'] == 'car' else (0, 255, 0) if track['type'] == 'truck' else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(frame, f"{track['type']} {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset = 30
        for v_type, counts in self.vehicle_counts.items():
            cv2.putText(frame, f"{v_type}: IN {counts['IN']} OUT {counts['OUT']}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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
            
            # Check if we need to generate a daily report
            current_date = datetime.now().date()
            if current_date != self.last_report_date:
                self.last_report_date = current_date
                self.gui_callback(None, "GENERATE_REPORT", "", "")
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

class VehicleCounterGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("YOLO Vehicle Counter Log - SQLite")
        self.logo_path = self._find_totalenergies_logo()
        self.init_sqlite()
        
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview with scrollbars
        tree_frame = tk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.tree = ttk.Treeview(tree_frame, columns=("ID", "Type", "Date", "Direction", "Check In", "Check Out"), show='headings')
        
        # Configure columns
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor='center', width=100)
        
        # Add scrollbars
        y_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        x_scroll = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        
        # Grid layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Filter frame
        filter_frame = tk.Frame(main_frame)
        filter_frame.pack(fill=tk.X, pady=(10, 5))
        
        # Date range filters
        tk.Label(filter_frame, text="From Date:").pack(side=tk.LEFT, padx=5)
        self.date_from_var = tk.StringVar()
        self.date_from_entry = tk.Entry(filter_frame, textvariable=self.date_from_var, width=12)
        self.date_from_entry.pack(side=tk.LEFT)
        
        tk.Label(filter_frame, text="To Date:").pack(side=tk.LEFT, padx=5)
        self.date_to_var = tk.StringVar()
        self.date_to_entry = tk.Entry(filter_frame, textvariable=self.date_to_var, width=12)
        self.date_to_entry.pack(side=tk.LEFT)
        
        # Vehicle type filter
        tk.Label(filter_frame, text="Vehicle Type:").pack(side=tk.LEFT, padx=5)
        self.vehicle_type_var = tk.StringVar(value="All")
        type_options = ["All"] + list(VEHICLE_CLASSES.values())
        self.type_dropdown = ttk.Combobox(filter_frame, textvariable=self.vehicle_type_var, 
                                        values=type_options, state="readonly", width=12)
        self.type_dropdown.pack(side=tk.LEFT)
        
        # Direction filter
        tk.Label(filter_frame, text="Direction:").pack(side=tk.LEFT, padx=5)
        self.direction_var = tk.StringVar(value="All")
        dir_options = ["All", "IN", "OUT"]
        self.direction_dropdown = ttk.Combobox(filter_frame, textvariable=self.direction_var, 
                                             values=dir_options, state="readonly", width=8)
        self.direction_dropdown.pack(side=tk.LEFT)
        
        # Apply filters button
        tk.Button(filter_frame, text="Apply Filters", command=self.load_from_sqlite).pack(side=tk.LEFT, padx=10)
        
        # Button frame
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        # Action buttons
        buttons = [
            ("Save to SQLite", self.save_to_sqlite),
            ("Load from SQLite", self.load_from_sqlite),
            ("Export to Excel", self.export_to_excel),
            ("Plot Chart", self.plot_chart),
            ("Export Chart", self.export_chart_to_image),
            ("Export PDF Report", self.export_pdf_report),
            ("Print All Records", self.print_all_records),
            ("Clear All Records", self.clear_all_records)
        ]
        
        for text, command in buttons:
            tk.Button(btn_frame, text=text, command=command).pack(side=tk.LEFT, padx=5, expand=True)
        
        # Initialize counter
        self.counter = YOLOVehicleCounter(self.add_or_update_entry)
        
        # Start video processing in separate thread
        threading.Thread(target=self.counter.run, daemon=True).start()
        
        # Set up window closing handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start periodic table refresh
        self.root.after(1000, self.refresh_table)
        
        # Load initial data
        self.load_from_sqlite()
        
        self.root.mainloop()

    def _find_totalenergies_logo(self):
        # Look for the logo in common locations
        possible_paths = [
            "totalenergies_logo.png",
            "totalenergies.png",
            "logo.png",
            os.path.expanduser("~/Downloads/totalenergies_logo.png"),
            os.path.expanduser("~/Pictures/totalenergies_logo.png"),
            os.path.expanduser("~/Documents/totalenergies_logo.png"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # If not found, ask user to select the logo
        messagebox.showinfo("Logo Required", "Please select the TotalEnergies logo image file.")
        logo_path = filedialog.askopenfilename(
            title="Select TotalEnergies Logo",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        
        if not logo_path:
            messagebox.showwarning("No Logo", "No logo selected. Reports will be generated without the company logo.")
        
        return logo_path if logo_path else None

    def init_sqlite(self):
        db_path = os.path.expanduser("~/Documents/vehicle_data.db")
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
            # Clear the treeview first
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Build the base query
            query = """
                SELECT vehicle_id, vehicle_type, date, direction, 
                       check_in_time, check_out_time 
                FROM vehicle_logs
            """
            
            filters = []
            params = []
            
            # Apply vehicle type filter if not "All"
            selected_type = self.vehicle_type_var.get()
            if selected_type and selected_type != "All":
                filters.append("LOWER(vehicle_type) = ?")
                params.append(selected_type.lower())
            
            # Apply direction filter if not "All"
            selected_dir = self.direction_var.get()
            if selected_dir and selected_dir != "All":
                filters.append("UPPER(direction) = ?")
                params.append(selected_dir.upper())
            
            # Apply date range filters if provided
            date_from = self.date_from_var.get().strip()
            date_to = self.date_to_var.get().strip()
            
            if date_from:
                try:
                    datetime.strptime(date_from, "%Y-%m-%d")
                    filters.append("date >= ?")
                    params.append(date_from)
                except ValueError:
                    messagebox.showerror("Invalid Date", "From Date must be in YYYY-MM-DD format")
                    return
                    
            if date_to:
                try:
                    datetime.strptime(date_to, "%Y-%m-%d")
                    filters.append("date <= ?")
                    params.append(date_to)
                except ValueError:
                    messagebox.showerror("Invalid Date", "To Date must be in YYYY-MM-DD format")
                    return
            
            # Combine filters if any exist
            if filters:
                query += " WHERE " + " AND ".join(filters)
            
            # Add sorting and limit
            query += " ORDER BY timestamp DESC LIMIT 100"
            
            # Execute the query
            self.sqlite_cursor.execute(query, params)
            rows = self.sqlite_cursor.fetchall()
            
            # Populate the treeview
            for row in rows:
                self.tree.insert("", "end", values=row)
                
            # Show message if no results
            if not rows:
                messagebox.showinfo("No Data", "No records match your filter criteria")
                
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to load data: {str(e)}")
            print(f"Error executing query: {e}")

    def export_to_excel(self):
        try:
            data = [self.tree.item(child)["values"] for child in self.tree.get_children()]
            if not data:
                messagebox.showinfo("No Data", "No records to export.")
                return
            df = pd.DataFrame(data, columns=["ID", "Type", "Date", "Direction", "Check In", "Check Out"])
            
            # Ask user for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                title="Save Excel File As",
                initialfile=f"vehicle_report_{datetime.now().strftime('%Y%m%d')}.xlsx"
            )
            
            if filename:
                df.to_excel(filename, index=False)
                messagebox.showinfo("Success", f"Data exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export to Excel: {str(e)}")

    def export_pdf_report(self):
        """Generate a professional PDF report with company logo and proper formatting"""
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            from matplotlib import font_manager as fm
            import os.path
            
            # Set font properties for the entire report
            font_props = {
                'title': {'fontname': 'Times New Roman', 'size': 24, 'weight': 'bold'},
                'header': {'fontname': 'Times New Roman', 'size': 16, 'weight': 'bold'},
                'subheader': {'fontname': 'Times New Roman', 'size': 14},
                'body': {'fontname': 'Times New Roman', 'size': 12},
                'small': {'fontname': 'Times New Roman', 'size': 10}
            }

            # Check if we have any data to report
            if not self.counter.vehicle_counts:
                messagebox.showwarning("No Data", "No vehicle data available to generate report.")
                return None

            # Ask user for save location
            report_date = datetime.now().strftime('%Y-%m-%d')
            filename = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                title="Save PDF Report As",
                initialfile=f"Vehicle_Report_{report_date}.pdf"
            )
            
            if not filename:
                return None  # User cancelled

            # Create PDF document
            with PdfPages(filename) as pdf:
                # Page 1: Cover Page
                fig_cover = plt.figure(figsize=(8.5, 11))
                ax_cover = fig_cover.add_axes([0, 0, 1, 1])
                ax_cover.axis('off')
                
                # Add company logo if available
                if self.logo_path:
                    try:
                        img = plt.imread(self.logo_path)
                        img_height = 1.5  # inches
                        img_width = img_height * (img.shape[1] / img.shape[0])
                        logo_ax = fig_cover.add_axes([0.5 - img_width/2, 0.7, img_width, img_height])
                        logo_ax.imshow(img)
                        logo_ax.axis('off')
                    except Exception as e:
                        print(f"Could not load logo: {e}")

                # Add report title and metadata
                ax_cover.text(0.5, 0.6, "VEHICLE TRAFFIC REPORT", 
                            ha='center', va='center', **font_props['title'])
                ax_cover.text(0.5, 0.5, "Daily Traffic Analysis Summary", 
                            ha='center', va='center', **font_props['subheader'])
                ax_cover.text(0.5, 0.4, f"Date: {report_date}", 
                            ha='center', va='center', **font_props['body'])
                
                pdf.savefig(fig_cover, bbox_inches='tight')
                plt.close(fig_cover)

                # Page 2: Executive Summary
                fig_summary = plt.figure(figsize=(8.5, 11))
                ax_summary = fig_summary.add_axes([0.1, 0.1, 0.8, 0.8])
                ax_summary.axis('off')
                
                # Summary title
                ax_summary.text(0.5, 0.95, "EXECUTIVE SUMMARY", 
                              ha='center', va='center', **font_props['header'])

                # Generate summary content
                summary_content = self._generate_summary_content()
                
                # Add summary text with proper formatting
                y_position = 0.85
                for section in summary_content:
                    # Add section header
                    ax_summary.text(0.1, y_position, section['title'], 
                                  **font_props['subheader'])
                    y_position -= 0.05
                    
                    # Add section content
                    for line in section['content']:
                        ax_summary.text(0.15, y_position, line, **font_props['body'])
                        y_position -= 0.04
                    
                    # Add spacing between sections
                    y_position -= 0.03
                
                pdf.savefig(fig_summary, bbox_inches='tight')
                plt.close(fig_summary)

                # Page 3: Data Visualization
                fig_chart = plt.figure(figsize=(8.5, 5))
                self._create_traffic_chart(fig_chart, font_props)
                pdf.savefig(fig_chart, bbox_inches='tight')
                plt.close(fig_chart)

                # Page 4: Detailed Data
                fig_data = plt.figure(figsize=(8.5, 11))
                ax_data = fig_data.add_axes([0.1, 0.1, 0.8, 0.8])
                ax_data.axis('off')
                
                if self.tree.get_children():
                    # Get data from treeview
                    data = []
                    for child in self.tree.get_children():
                        data.append(self.tree.item(child)["values"])
                    
                    # Create a table
                    columns = ["ID", "Type", "Date", "Direction", "Check In", "Check Out"]
                    table = ax_data.table(cellText=data, colLabels=columns, 
                                         cellLoc='center', loc='center')
                    table.auto_set_font_size(False)
                    table.set_fontsize(8)
                    table.scale(1, 1.2)
                    
                    # Add title
                    ax_data.text(0.5, 0.95, "DETAILED TRAFFIC DATA", 
                                ha='center', va='center', **font_props['header'])
                else:
                    ax_data.text(0.5, 0.5, "No detailed traffic data available", 
                               ha='center', va='center', **font_props['body'])
                
                pdf.savefig(fig_data, bbox_inches='tight')
                plt.close(fig_data)

            # Show success message
            messagebox.showinfo(
                "Report Generated",
                f"PDF report successfully created:\n\n{filename}"
            )
            return filename

        except Exception as e:
            error_msg = (
                "Failed to generate PDF report.\n\n"
                f"Error: {str(e)}\n\n"
                "Please ensure:\n"
                "- You have write permissions\n"
                "- The document isn't open in another program\n"
                "- There's enough disk space"
            )
            messagebox.showerror("Report Generation Failed", error_msg)
            return None

    def _generate_summary_content(self):
        """Generate structured content for the executive summary"""
        vehicle_types = list(self.counter.vehicle_counts.keys())
        in_counts = [self.counter.vehicle_counts[vt]['IN'] for vt in vehicle_types]
        out_counts = [self.counter.vehicle_counts[vt]['OUT'] for vt in vehicle_types]
        
        total_in = sum(in_counts)
        total_out = sum(out_counts)
        net_change = total_in - total_out
        
        # Get peak hour if data available
        peak_hour_info = ""
        try:
            check_in_times = [
                self.tree.item(child)["values"][4] 
                for child in self.tree.get_children() 
                if self.tree.item(child)["values"][4]
            ]
            if check_in_times:
                hours = [datetime.strptime(t, "%H:%M:%S").hour for t in check_in_times]
                peak_hour = max(set(hours), key=hours.count)
                peak_hour_info = f"Peak traffic hour: {peak_hour}:00 - {peak_hour+1}:00"
        except:
            pass
        
        return [
            {
                'title': "TOTAL TRAFFIC",
                'content': [
                    f"Inbound vehicles: {total_in}",
                    f"Outbound vehicles: {total_out}",
                    f"Net change: {net_change}",
                    peak_hour_info if peak_hour_info else ""
                ]
            },
            {
                'title': "TRAFFIC BY VEHICLE TYPE",
                'content': [
                    f"{vt}: IN {in_counts[i]} | OUT {out_counts[i]}" 
                    for i, vt in enumerate(vehicle_types)
                ]
            }
        ]

    def _create_traffic_chart(self, fig, font_props):
        """Create the traffic visualization chart"""
        ax = fig.add_subplot(111)
        vehicle_types = list(self.counter.vehicle_counts.keys())
        in_counts = [self.counter.vehicle_counts[vt]['IN'] for vt in vehicle_types]
        out_counts = [self.counter.vehicle_counts[vt]['OUT'] for vt in vehicle_types]
        
        x = range(len(vehicle_types))
        width = 0.35
        
        # Create bars with corporate colors
        ax.bar([i - width/2 for i in x], in_counts, width, 
              label='INBOUND', color='#004b8d')  # Corporate blue
        ax.bar([i + width/2 for i in x], out_counts, width, 
              label='OUTBOUND', color='#f04028')  # Corporate red
        
        # Format chart
        ax.set_xlabel('Vehicle Type', **font_props['body'])
        ax.set_ylabel('Number of Vehicles', **font_props['body'])
        ax.set_title('DAILY TRAFFIC BY VEHICLE TYPE', **font_props['header'])
        ax.set_xticks(x)
        ax.set_xticklabels(vehicle_types, **font_props['body'])
        
        # Apply Times New Roman to all text elements
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontname('Times New Roman')
        
        ax.legend(prop={'family': 'Times New Roman', 'size': 10})
        fig.tight_layout()

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
            
            # Ask user for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                title="Save Chart Image As",
                initialfile=f"vehicle_chart_{datetime.now().strftime('%Y%m%d')}.png"
            )
            
            if filename:
                fig.savefig(filename)
                messagebox.showinfo("Success", f"Chart exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export chart: {str(e)}")

    def plot_chart(self):
        try:
            vehicle_types = list(self.counter.vehicle_counts.keys())
            in_counts = [self.counter.vehicle_counts[vt]['IN'] for vt in vehicle_types]
            out_counts = [self.counter.vehicle_counts[vt]['OUT'] for vt in vehicle_types]
            
            # Create a new window for the chart
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Vehicle Count Chart")
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 5))
            x = range(len(vehicle_types))
            width = 0.35
            
            # Create bars
            ax.bar([i - width/2 for i in x], in_counts, width, label='IN', color='green')
            ax.bar([i + width/2 for i in x], out_counts, width, label='OUT', color='red')
            
            # Configure chart
            ax.set_xlabel('Vehicle Type')
            ax.set_ylabel('Count')
            ax.set_title('Vehicle Counts by Type and Direction')
            ax.set_xticks(x)
            ax.set_xticklabels(vehicle_types)
            ax.legend()
            
            # Embed in Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=plot_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = tk.Frame(plot_window)
            toolbar.pack(fill=tk.X)
            tk.Button(toolbar, text="Export Image", 
                     command=lambda: self.export_chart_to_image(fig)).pack(side=tk.LEFT)
            tk.Button(toolbar, text="Close", 
                     command=plot_window.destroy).pack(side=tk.RIGHT)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create plot: {str(e)}")

    def add_or_update_entry(self, vid, direction, time_str, vehicle_type):
        if direction == "GENERATE_REPORT":
            # Generate daily report
            self.export_pdf_report()
            return
            
        date_str = datetime.now().strftime("%Y-%m-%d")
        existing = None
        
        # Find existing entry if it exists
        for child in self.tree.get_children():
            try:
                vals = self.tree.item(child)["values"]
                if vals and vals[0] == vid:
                    existing = child
                    break
            except tk.TclError:
                continue  # Skip if item no longer exists
        
        if existing:
            try:
                vals = list(self.tree.item(existing)["values"])
                if direction == "IN":
                    vals[1:5] = [vehicle_type, date_str, direction, time_str]
                elif direction == "OUT":
                    vals[1:4] = [vehicle_type, date_str, direction]
                    vals[5] = time_str
                self.tree.item(existing, values=vals)
            except tk.TclError:
                # If item disappeared between checking and updating, create new entry
                if direction == "IN":
                    self.tree.insert("", "end", values=(vid, vehicle_type, date_str, direction, time_str, ""))
                else:
                    self.tree.insert("", "end", values=(vid, vehicle_type, date_str, direction, "", time_str))
        else:
            if direction == "IN":
                self.tree.insert("", "end", values=(vid, vehicle_type, date_str, direction, time_str, ""))
            else:
                self.tree.insert("", "end", values=(vid, vehicle_type, date_str, direction, "", time_str))

    def refresh_table(self):
        for vid, rec in self.counter.vehicle_records.copy().items():
            existing = None
            for child in self.tree.get_children():
                try:
                    vals = self.tree.item(child)["values"]
                    if vals and vals[0] == vid:
                        existing = child
                        break
                except tk.TclError:
                    continue
            
            if existing:
                try:
                    vals = list(self.tree.item(existing)["values"])
                    vals[1] = rec.get("vehicle_type", "")
                    vals[2] = rec.get("date", "")
                    vals[3] = rec.get("direction", "")
                    vals[4] = rec.get("check_in_time", "") or ""
                    vals[5] = rec.get("check_out_time", "") or ""
                    self.tree.item(existing, values=vals)
                except tk.TclError:
                    # If item disappeared, create new entry
                    self.tree.insert("", "end", values=(
                        vid,
                        rec.get("vehicle_type", ""),
                        rec.get("date", ""),
                        rec.get("direction", ""),
                        rec.get("check_in_time", "") or "",
                        rec.get("check_out_time", "") or ""
                    ))
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