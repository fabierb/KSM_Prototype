import tkinter as tk
from tkinter import ttk
import time
import serial
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import re
import csv
import os

# ----- Configuration Constants -----
POLLING_INTERVAL = 0.25       # seconds
# Default serial port; adjust as needed (e.g. "COM6" on Windows)
DEFAULT_SERIAL_PORT = "COM6"
DEFAULT_BAUDRATE = 9600               # default baud rate
DEFAULT_MUX_ID = "001"                 # default MUX identifier
COMMAND_TERMINATOR = "\r"             # command termination character

# Fixed bin dimensions
BIN_LENGTH = 0.65  # meters
BIN_WIDTH = 0.45   # meters
# ------------------------------------

def calculate_checksum(command: str) -> str:
    checksum = 0
    for char in command[0:]:
        checksum ^= ord(char)
    return f"{checksum:02X}"

def parse_scale_token(token: str):
    """Parse '@LLswwwwwwwwx' token into scale index, weight and status."""
    if not token.startswith('@') or len(token) < 13:
        return None
    try:
        scale = int(token[3])
        weight = int(token[4:12]) / 1000.0
        status = token[12]
        return scale, weight, status
    except Exception:
        return None

def parse_mux_weights(response: str):
    """Extract the four scale weights from a raw response string."""
    if response.startswith('@'):
        response = response[1:]
    # Drop leading identifier digits (e.g. "69") if present
    if len(response) > 2 and response[:2].isdigit():
        response = response[2:]

    matches = re.findall(r'([+-]?\d+\.\d+)([A-Za-z]?)', response)
    weights = []
    statuses = []
    for value, status in matches:
        try:
            weights.append(float(value))
            statuses.append(status)
        except ValueError:
            continue
    return weights[:4], statuses[:4]

class ScaleMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("Scale Monitor")
        
        # Serial connection variables for scale MUX
        self.serial_port = tk.StringVar(value=DEFAULT_SERIAL_PORT)
        self.baud_rate = tk.IntVar(value=DEFAULT_BAUDRATE)
        self.mux_id = tk.StringVar(value=DEFAULT_MUX_ID)
        self.ser = None
        self.connect_serial()
        
        # Display variables
        self.weights = [tk.StringVar(value="0.0 kg") for _ in range(4)]
        self.total_weight = tk.StringVar(value="0.0 kg")
        self.object_position = tk.StringVar(value="N/A")
        
        # Tare values
        self.tare_active = False
        self.tare_values = [0.0, 0.0, 0.0, 0.0]
        self.last_raw_weights = [0.0, 0.0, 0.0, 0.0]
        
        # (Filtering removed)
        
        # Rounding resolution: determined dynamically by a percentage of the object unit weight.
        self.rounding_percentage_var = tk.DoubleVar(value=10)  # default 10%
        
        # Remove expected weight change functionality.
        # self.expected_weight_change is removed along with self.previous_total_weight.
        
        # Object counting
        self.object_unit_weight = 0.01
        self.object_count = 0
        
        # Auto-tare parameters
        self.auto_tare_threshold_var = tk.DoubleVar(value=4)
        self.auto_tare_cooldown_var = tk.DoubleVar(value=1.5)
        self.auto_tare_stability_time_var = tk.DoubleVar(value=1.0)
        self.auto_tare_stability_std_threshold_var = tk.DoubleVar(value=0.01)
        self.last_auto_tare_time = 0.0
        self.total_weight_history = []
        
        # Scale positions (assumed at corners with slight offsets)
        self.scale_positions = [
            (0.1, BIN_WIDTH - 0.1),
            (BIN_LENGTH - 0.1, BIN_WIDTH - 0.1),
            (0.1, 0.1),
            (BIN_LENGTH - 0.1, 0.1)
        ]
        
        # Compartments Layout Parameters (set via GUI)
        self.num_cols = tk.IntVar(value=4)  # default columns
        self.num_rows = tk.IntVar(value=2)  # default rows
        
        # Expected Compartment Section
        self.expected_compartment = tk.IntVar(value=1)
        # Expected Object Count Section
        self.expected_count = tk.IntVar(value=1)
        
        # Pick Status display (using tk.Label to allow background color changes)
        self.pick_status_text = tk.StringVar(value="Pick Status: N/A")
        self.pick_status_label = tk.Label(textvariable=self.pick_status_text, bg="grey", font=("Arial", 10, "bold"))
        
        # Certainty variables: weight, location, composite.
        self.weight_certainty_text = tk.StringVar(value="Weight Certainty: N/A")
        self.location_certainty_text = tk.StringVar(value="Location Certainty: N/A")
        self.certainty_text = tk.StringVar(value="Composite Certainty: N/A")
        self.composite_certainty = None
        
        # Certainty Threshold for pick confirmation (in percent)
        self.certainty_threshold_var = tk.DoubleVar(value=80)  # default 80%
        
        # Graph data arrays
        self.time_data = np.array([])
        self.weight_data = [np.array([]) for _ in range(4)]
        self.total_weight_data = np.array([])
        self.start_time = time.time()

        # Directory to store CSV logs
        self.log_dir = "logs"

        self.running = True
        self.create_ui()
        # Schedule polling on the Tk event loop instead of using a thread
        self.root.after(int(POLLING_INTERVAL * 1000), self.poll_weights)

    def connect_serial(self):
        try:
            if self.ser:
                self.ser.close()
        except Exception as e:
            print("Error closing serial port:", e)
        port = self.serial_port.get() if isinstance(self.serial_port, tk.Variable) else self.serial_port
        baud = self.baud_rate.get() if isinstance(self.baud_rate, tk.Variable) else self.baud_rate
        try:
            self.ser = serial.Serial(port, baudrate=baud, timeout=1)
            print(f"Connected to serial port {port} at {baud} baud")
        except Exception as e:
            print(f"Failed to open serial port {port}: {e}")
    
    def create_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=True, fill="both")
        control_tab = ttk.Frame(notebook)
        notebook.add(control_tab, text="Control")
        self.create_control_ui(control_tab)
    
    def create_control_ui(self, parent):
        # Divide window into left (controls) and right (graphs)
        left_frame = ttk.Frame(parent)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        right_frame = ttk.Frame(parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- Left Column (Controls) ---
        serial_frame = ttk.LabelFrame(left_frame, text="Serial Connection", padding=10)
        serial_frame.pack(fill=tk.X, pady=5)
        ttk.Label(serial_frame, text="Port:").pack(side=tk.LEFT)
        self.serial_entry = ttk.Entry(serial_frame, textvariable=self.serial_port, width=10)
        self.serial_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(serial_frame, text="Baud:").pack(side=tk.LEFT)
        self.baud_entry = ttk.Entry(serial_frame, textvariable=self.baud_rate, width=6)
        self.baud_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(serial_frame, text="MUX ID:").pack(side=tk.LEFT)
        self.mux_entry = ttk.Entry(serial_frame, textvariable=self.mux_id, width=4)
        self.mux_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(serial_frame, text="Connect", command=self.connect_serial).pack(side=tk.LEFT, padx=5)
        
        # Scale Info Section (including Total Weight)
        scale_info = ttk.LabelFrame(left_frame, text="Scale Info", padding=10)
        scale_info.pack(fill=tk.X, pady=5)
        for i in range(4):
            frm = ttk.Frame(scale_info)
            frm.pack(fill=tk.X, pady=2)
            ttk.Label(frm, text=f"Scale {i+1}:", width=10).pack(side=tk.LEFT)
            ttk.Label(frm, textvariable=self.weights[i], width=10).pack(side=tk.LEFT)
            ttk.Button(frm, text="Zero", command=lambda i=i: self.zero_scale(i)).pack(side=tk.LEFT, padx=5)
        total_frm = ttk.Frame(scale_info)
        total_frm.pack(fill=tk.X, pady=2)
        ttk.Label(total_frm, text="Total Weight:", width=12, font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        ttk.Label(total_frm, textvariable=self.total_weight, width=10, font=("Arial", 10)).pack(side=tk.LEFT)
        
        # Tare & Reset Section
        tare_reset = ttk.LabelFrame(left_frame, text="Tare & Reset", padding=10)
        tare_reset.pack(fill=tk.X, pady=5)
        ttk.Button(tare_reset, text="Tare Bin", command=self.tare_bin).pack(side=tk.LEFT, padx=5)
        ttk.Button(tare_reset, text="Clear Tare", command=self.clear_tare).pack(side=tk.LEFT, padx=5)
        ttk.Button(tare_reset, text="Reset Graphs", command=self.reset_graphs).pack(side=tk.LEFT, padx=5)
        
        # Object Unit Weight Section
        unit_weight = ttk.LabelFrame(left_frame, text="Object Unit Weight", padding=10)
        unit_weight.pack(fill=tk.X, pady=5)
        ttk.Label(unit_weight, text="Weight (kg):").pack(side=tk.LEFT)
        self.object_weight_entry = ttk.Entry(unit_weight, width=10)
        self.object_weight_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(unit_weight, text="Set", command=self.set_object_unit_weight).pack(side=tk.LEFT, padx=5)
        
        # Rounding Resolution (dynamic)
        rounding = ttk.LabelFrame(left_frame, text="Rounding", padding=10)
        rounding.pack(fill=tk.X, pady=5)
        ttk.Label(rounding, text="Rounding % of Object Unit:").pack(side=tk.LEFT)
        self.rounding_percentage_var = tk.DoubleVar(value=10)
        ttk.Entry(rounding, textvariable=self.rounding_percentage_var, width=7).pack(side=tk.LEFT, padx=5)
        ttk.Label(rounding, text="(resolution = unit_weight * (%/100))").pack(side=tk.LEFT)
        
        # (Expected Weight Change section removed)
        
        # Compartments Layout Section
        comp_layout = ttk.LabelFrame(left_frame, text="Compartments Layout", padding=10)
        comp_layout.pack(fill=tk.X, pady=5)
        ttk.Label(comp_layout, text="Columns:").pack(side=tk.LEFT)
        self.num_cols = tk.IntVar(value=4)
        ttk.Entry(comp_layout, textvariable=self.num_cols, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(comp_layout, text="Rows:").pack(side=tk.LEFT)
        self.num_rows = tk.IntVar(value=2)
        ttk.Entry(comp_layout, textvariable=self.num_rows, width=5).pack(side=tk.LEFT, padx=5)
        
        # Expected Compartment Section
        exp_comp = ttk.LabelFrame(left_frame, text="Expected Compartment", padding=10)
        exp_comp.pack(fill=tk.X, pady=5)
        ttk.Label(exp_comp, text="Expected #:").pack(side=tk.LEFT)
        self.expected_compartment = tk.IntVar(value=1)
        ttk.Spinbox(exp_comp, from_=1, to=100, textvariable=self.expected_compartment, width=5).pack(side=tk.LEFT, padx=5)
        
        # Expected Object Count Section
        exp_count = ttk.LabelFrame(left_frame, text="Expected Object Count", padding=10)
        exp_count.pack(fill=tk.X, pady=5)
        ttk.Label(exp_count, text="Expected Count:").pack(side=tk.LEFT)
        self.expected_count = tk.IntVar(value=1)
        ttk.Spinbox(exp_count, from_=0, to=100, textvariable=self.expected_count, width=5).pack(side=tk.LEFT, padx=5)
        
        # Pick Status display
        pick_status_frame = ttk.LabelFrame(left_frame, text="Pick Status", padding=10)
        pick_status_frame.pack(fill=tk.X, pady=5)
        self.pick_status_label = tk.Label(pick_status_frame, textvariable=self.pick_status_text, bg="grey", font=("Arial", 10, "bold"))
        self.pick_status_label.pack(side=tk.LEFT, fill=tk.X)
        
        # Certainty Section
        certainty_frame = ttk.LabelFrame(left_frame, text="Certainty", padding=10)
        certainty_frame.pack(fill=tk.X, pady=5)
        ttk.Label(certainty_frame, textvariable=self.weight_certainty_text).pack(fill=tk.X)
        ttk.Label(certainty_frame, textvariable=self.location_certainty_text).pack(fill=tk.X)
        ttk.Label(certainty_frame, textvariable=self.certainty_text).pack(fill=tk.X)
        
        # Certainty Threshold for Pick Confirmation
        thresh_frame = ttk.LabelFrame(left_frame, text="Certainty Threshold", padding=10)
        thresh_frame.pack(fill=tk.X, pady=5)
        ttk.Label(thresh_frame, text="Min Composite Certainty (%):").pack(side=tk.LEFT)
        self.certainty_threshold_var = tk.DoubleVar(value=80)
        ttk.Entry(thresh_frame, textvariable=self.certainty_threshold_var, width=7).pack(side=tk.LEFT, padx=5)
        
        # --- Right Column (Graphs) ---
        graphs_frame = ttk.Frame(right_frame)
        graphs_frame.pack(fill=tk.BOTH, expand=True)
        
        # Place graphs side by side: one for time-series and one for position-based graphs.
        ts_frame = ttk.Frame(graphs_frame)
        ts_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        pos_frame = ttk.Frame(graphs_frame)
        pos_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Weight Changes Over Time")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Weight (kg)")
        self.canvas = FigureCanvasTkAgg(self.fig, master=ts_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig_total, self.ax_total = plt.subplots()
        self.ax_total.set_title("Total Weight Over Time")
        self.ax_total.set_xlabel("Time (s)")
        self.ax_total.set_ylabel("Total Weight (kg)")
        self.canvas_total = FigureCanvasTkAgg(self.fig_total, master=ts_frame)
        self.canvas_total.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig_position, self.ax_position = plt.subplots()
        self.ax_position.set_title("Object Position")
        self.ax_position.set_xlabel("Length (m)")
        self.ax_position.set_ylabel("Width (m)")
        self.ax_position.set_xlim(0, BIN_LENGTH)
        self.ax_position.set_ylim(0, BIN_WIDTH)
        self.ax_position.set_aspect('equal', adjustable='box')
        self.canvas_position = FigureCanvasTkAgg(self.fig_position, master=pos_frame)
        self.canvas_position.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig_compartment, self.ax_compartment = plt.subplots()
        self.ax_compartment.set_title("Compartment Layout")
        self.ax_compartment.set_xlabel("Length (m)")
        self.ax_compartment.set_ylabel("Width (m)")
        self.ax_compartment.set_xlim(0, BIN_LENGTH)
        self.ax_compartment.set_ylim(0, BIN_WIDTH)
        self.ax_compartment.set_aspect('equal', adjustable='box')
        self.canvas_compartment = FigureCanvasTkAgg(self.fig_compartment, master=pos_frame)
        self.canvas_compartment.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def calculate_center_of_mass(self, weights):
        total = sum(weights)
        if abs(total) < 1e-6:
            return None, None
        x_cm = (weights[0]*self.scale_positions[0][0] +
                weights[1]*self.scale_positions[1][0] +
                weights[2]*self.scale_positions[2][0] +
                weights[3]*self.scale_positions[3][0]) / total
        y_cm = (weights[0]*self.scale_positions[0][1] +
                weights[1]*self.scale_positions[1][1] +
                weights[2]*self.scale_positions[2][1] +
                weights[3]*self.scale_positions[3][1]) / total
        return x_cm, y_cm
        
    def set_object_unit_weight(self):
        try:
            weight = float(self.object_weight_entry.get())
            if weight > 0:
                self.object_unit_weight = weight
                print(f"Object unit weight set to {self.object_unit_weight} kg")
            else:
                print("Please enter a positive weight.")
        except ValueError:
            print("Invalid entry for object weight.")

    def log_measurements(self):
        """Write the collected measurement data to a CSV file."""
        if len(self.time_data) == 0:
            print("No measurement data to log.")
            return
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.log_dir, f"weights_{timestamp}.csv")
        header = ["time"] + [f"scale{i+1}" for i in range(4)] + ["total"]
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for idx, t in enumerate(self.time_data):
                row = [t]
                for i in range(4):
                    if idx < len(self.weight_data[i]):
                        row.append(self.weight_data[i][idx])
                    else:
                        row.append("")
                if idx < len(self.total_weight_data):
                    row.append(self.total_weight_data[idx])
                else:
                    row.append("")
                writer.writerow(row)
        print(f"Measurement log saved to {filename}")
    
    def reset_graphs(self):
        # Save the existing measurement data before clearing
        self.log_measurements()

        self.time_data = np.array([])
        self.weight_data = [np.array([]) for _ in range(4)]
        self.total_weight_data = np.array([])
        self.start_time = time.time()
        
        self.ax.clear()
        self.ax.set_title("Weight Changes Over Time")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Weight (kg)")
        self.canvas.draw()
        
        self.ax_total.clear()
        self.ax_total.set_title("Total Weight Over Time")
        self.ax_total.set_xlabel("Time (s)")
        self.ax_total.set_ylabel("Total Weight (kg)")
        self.canvas_total.draw()
        
        self.ax_position.clear()
        self.ax_position.set_title("Object Position")
        self.ax_position.set_xlabel("Length (m)")
        self.ax_position.set_ylabel("Width (m)")
        self.ax_position.set_xlim(0, BIN_LENGTH)
        self.ax_position.set_ylim(0, BIN_WIDTH)
        self.ax_position.set_aspect('equal', adjustable='box')
        self.canvas_position.draw()
        
        self.clear_compartment_display()
    
    def zero_scale(self, scale_index):
        print(f"Attempting to zero Scale {scale_index+1}")
        # Zeroing not implemented for serial connection
    
    def tare_bin(self):
        if len(self.last_raw_weights) == 4:
            self.tare_values = self.last_raw_weights.copy()
            self.tare_active = True
            print("Tare set. Baseline values:", self.tare_values)
        else:
            print("No valid readings available to tare.")
    
    def clear_tare(self):
        self.tare_active = False
        self.tare_values = [0.0, 0.0, 0.0, 0.0]
        print("Tare cleared.")
    
    def poll_weights(self):
        if self.running:
            self.request_weights()
            self.root.after(int(POLLING_INTERVAL * 1000), self.poll_weights)
    
    def request_weights(self):
        try:
            if not self.ser:
                return

            mux = self.mux_id.get() if isinstance(self.mux_id, tk.Variable) else self.mux_id
            cmd = f"@08gl{mux}"
            checksum = calculate_checksum(cmd)
            full_cmd = f"{cmd}{checksum}{COMMAND_TERMINATOR}"
            raw_cmd = full_cmd.encode("ascii")
            print(f"Sending: {full_cmd}")
            self.ser.write(raw_cmd)

            raw_response = self.ser.readline()
            print(f"Raw Response: {raw_response}")
            response = raw_response.decode("latin-1", errors="replace").strip()
            if not response:
                return
            print(f"Decoded Response: {response}")

            scale_values, statuses = parse_mux_weights(response)
            if len(scale_values) < 4:
                print("Incomplete response from scale MUX")
                return

            current_time = time.time() - self.start_time
            self.time_data = np.append(self.time_data, current_time)

            total_weight = 0.0
            raw_weights = []
            adjusted_weights = []
            for i in range(4):
                raw_value = float(scale_values[i])
                perc = self.rounding_percentage_var.get() / 100.0
                if self.object_unit_weight > 0:
                    rounding_factor = self.object_unit_weight * perc
                else:
                    rounding_factor = 0.01
                rounded_value = round(raw_value / rounding_factor) * rounding_factor

                value_to_use = rounded_value
                raw_weights.append(raw_value)
                if self.tare_active:
                    adjusted = value_to_use - self.tare_values[i]
                else:
                    adjusted = value_to_use
                adjusted_weights.append(adjusted)
                self.weights[i].set(f"{adjusted:.3f} kg")
                total_weight += adjusted
                if len(self.weight_data[i]) < len(self.time_data):
                    self.weight_data[i] = np.append(self.weight_data[i], adjusted)

            self.last_raw_weights = raw_weights.copy()
            self.total_weight.set(f"{total_weight:.3f} kg")
            self.total_weight_data = np.append(self.total_weight_data, total_weight)
            self.update_graphs_data()

            # (Expected weight change update removed)
            self.previous_total_weight = total_weight

            # --- Certainty Calculation ---
            window_size = 5
            if len(self.total_weight_data) >= window_size:
                recent_weights = self.total_weight_data[-window_size:]
                std_weight = np.std(recent_weights)
                expected_weight = self.object_unit_weight if self.object_unit_weight > 0 else 0.01
                weight_certainty = max(0, 100 * (1 - (std_weight / expected_weight)))
                self.weight_certainty_text.set(f"Weight Certainty: {weight_certainty:.1f}%")
            else:
                self.weight_certainty_text.set("Weight Certainty: N/A")

            # Location Certainty: Compare computed COM to the expected compartment center.
            com = self.calculate_center_of_mass(adjusted_weights)
            if com[0] is not None and com[1] is not None:
                num_cols = self.num_cols.get()
                num_rows = self.num_rows.get()
                col_width = BIN_LENGTH / num_cols
                row_height = BIN_WIDTH / num_rows
                exp = self.expected_compartment.get()
                exp_index = exp - 1
                exp_row = exp_index // num_cols
                exp_col = exp_index % num_cols
                expected_center_x = exp_col * col_width + col_width / 2
                expected_center_y = exp_row * row_height + row_height / 2
                dist = np.sqrt((com[0] - expected_center_x)**2 + (com[1] - expected_center_y)**2)
                threshold_loc = min(col_width, row_height) / 2
                location_certainty = max(0, 100 * (1 - (dist / threshold_loc)))
                self.location_certainty_text.set(f"Location Certainty: {location_certainty:.1f}%")
            else:
                self.location_certainty_text.set("Location Certainty: N/A")

            if (len(self.total_weight_data) >= window_size and com[0] is not None):
                composite_certainty = (weight_certainty + location_certainty) / 2.0
                self.certainty_text.set(f"Composite Certainty: {composite_certainty:.1f}%")
                self.composite_certainty = composite_certainty
            else:
                self.certainty_text.set("Composite Certainty: N/A")
                self.composite_certainty = None

            # Auto-tare based on absolute conditions
            current_timestamp = time.time()
            self.total_weight_history.append((current_timestamp, total_weight))
            self.total_weight_history = [(t, w) for (t, w) in self.total_weight_history
                                         if current_timestamp - t <= self.auto_tare_stability_time_var.get()]
            if len(self.total_weight_history) >= window_size:
                recent_weights_full = np.array([w for (t, w) in self.total_weight_history])
                stdev = np.std(recent_weights_full)
            else:
                stdev = float('inf')

            if (abs(total_weight) > self.auto_tare_threshold_var.get() and
                stdev < self.auto_tare_stability_std_threshold_var.get() and
                (current_timestamp - self.last_auto_tare_time) > self.auto_tare_cooldown_var.get()):
                print("Auto-tare triggered (weight stabilized above threshold)!")
                self.tare_bin()
                self.last_auto_tare_time = current_timestamp
                total_weight = 0.0

            # Update object count
            if self.object_unit_weight > 0:
                self.object_count = int(round(abs(total_weight) / self.object_unit_weight))
            else:
                self.object_count = 0

            if len(adjusted_weights) == 4 and abs(total_weight) >= 0.01:
                x, y = self.calculate_center_of_mass(adjusted_weights)
                if x is not None and y is not None:
                    if total_weight < 0:
                        self.object_position.set(f"Removed from: X: {x:.3f} m, Y: {y:.3f} m")
                    else:
                        self.object_position.set(f"X: {x:.3f} m, Y: {y:.3f} m")
                    self.update_object_position_plot(x, y)
                    self.update_compartment_display(x, y)
                else:
                    self.object_position.set("N/A")
                    self.clear_object_position_plot()
                    self.clear_compartment_display()
            else:
                self.object_position.set("N/A")
                self.clear_object_position_plot()
                self.clear_compartment_display()
        except Exception as e:
            print("Error reading from scale:", e)
    
    def update_graphs_data(self):
        self.ax.clear()
        self.ax_total.clear()
        for i in range(4):
            if len(self.weight_data[i]) > 0:
                self.ax.plot(self.time_data, self.weight_data[i], label=f"Scale {i+1}")
        self.ax_total.plot(self.time_data, self.total_weight_data, label="Total Weight", color='red')
        self.ax.legend()
        self.ax_total.legend()
        self.canvas.draw()
        self.canvas_total.draw()
    
    def update_object_position_plot(self, x, y):
        self.ax_position.clear()
        rect = plt.Rectangle((0, 0), BIN_LENGTH, BIN_WIDTH, fill=False, edgecolor='black', linewidth=2)
        self.ax_position.add_patch(rect)
        self.ax_position.plot(x, y, 'ro', markersize=10)
        self.ax_position.set_xlim(0, BIN_LENGTH)
        self.ax_position.set_ylim(0, BIN_WIDTH)
        self.ax_position.set_title("Object Position")
        self.ax_position.set_xlabel("Length (m)")
        self.ax_position.set_ylabel("Width (m)")
        self.canvas_position.draw()
    
    def clear_object_position_plot(self):
        self.ax_position.clear()
        self.ax_position.set_title("Object Position")
        self.ax_position.set_xlabel("Length (m)")
        self.ax_position.set_ylabel("Width (m)")
        self.ax_position.set_xlim(0, BIN_LENGTH)
        self.ax_position.set_ylim(0, BIN_WIDTH)
        self.canvas_position.draw()
    
    def update_compartment_display(self, x, y):
        num_cols = self.num_cols.get()
        num_rows = self.num_rows.get()
        col_width = BIN_LENGTH / num_cols
        row_height = BIN_WIDTH / num_rows
        
        col = int(x / col_width)
        row = int(y / row_height)
        computed_compartment = row * num_cols + col + 1
        
        self.ax_compartment.clear()
        outer = plt.Rectangle((0, 0), BIN_LENGTH, BIN_WIDTH, fill=False, edgecolor='black', linewidth=2)
        self.ax_compartment.add_patch(outer)
        for c in range(1, num_cols):
            self.ax_compartment.plot([c * col_width, c * col_width], [0, BIN_WIDTH], color='black', linestyle='--')
        for r in range(1, num_rows):
            self.ax_compartment.plot([0, BIN_LENGTH], [r * row_height, r * row_height], color='black', linestyle='--')
        
        if self.object_count > 0:
            x0 = col * col_width
            y0 = row * row_height
            highlight = plt.Rectangle((x0, y0), col_width, row_height, color='lightgreen', alpha=0.5)
            self.ax_compartment.add_patch(highlight)
            x_center = x0 + col_width / 2
            y_center = y0 + row_height / 2
            self.ax_compartment.text(x_center, y_center, f"Count: {self.object_count}",
                                     horizontalalignment='center', verticalalignment='center',
                                     fontsize=14, color='blue')
        self.ax_compartment.set_xlim(0, BIN_LENGTH)
        self.ax_compartment.set_ylim(0, BIN_WIDTH)
        self.ax_compartment.set_title("Compartment Layout")
        self.ax_compartment.set_xlabel("Length (m)")
        self.ax_compartment.set_ylabel("Width (m)")
        self.canvas_compartment.draw()
        
        # Compare computed compartment, expected count, and composite certainty.
        expected_comp = self.expected_compartment.get()
        expected_count = self.expected_count.get()
        if (computed_compartment == expected_comp and 
            self.object_count == expected_count and 
            self.composite_certainty is not None and 
            self.composite_certainty >= self.certainty_threshold_var.get()):
            self.pick_status_text.set("Pick Status: Confirmed")
            self.pick_status_label.config(bg="green")
        else:
            self.pick_status_text.set("Pick Status: Not Confirmed")
            self.pick_status_label.config(bg="red")
    
    def clear_compartment_display(self):
        self.ax_compartment.clear()
        outer = plt.Rectangle((0, 0), BIN_LENGTH, BIN_WIDTH, fill=False, edgecolor='black', linewidth=2)
        self.ax_compartment.add_patch(outer)
        self.ax_compartment.plot([BIN_LENGTH/2, BIN_LENGTH/2], [0, BIN_WIDTH], color='black', linestyle='--')
        self.ax_compartment.plot([0, BIN_LENGTH], [BIN_WIDTH/2, BIN_WIDTH/2], color='black', linestyle='--')
        self.ax_compartment.set_xlim(0, BIN_LENGTH)
        self.ax_compartment.set_ylim(0, BIN_WIDTH)
        self.ax_compartment.set_title("Compartment Layout")
        self.ax_compartment.set_xlabel("Length (m)")
        self.ax_compartment.set_ylabel("Width (m)")
        self.canvas_compartment.draw()
    
    def simulate_object_addition(self):
        # Manual trigger: previously set expected weight change; now removed.
        print("Simulated object addition triggered (no expected weight change used).")
    
    def on_close(self):
        self.running = False
        self.root.destroy()
        
if __name__ == "__main__":
    root = tk.Tk()
    app = ScaleMonitor(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
