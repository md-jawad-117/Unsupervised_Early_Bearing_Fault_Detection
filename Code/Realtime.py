

import sys, os, re, time, json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from PyQt5 import QtCore, QtGui, QtWidgets

def fmt_time_ms(ms: float) -> str:
 
    if ms < 0.001:
        return f"{ms*1000:.1f} µs"
    if ms < 1:
        return f"{ms:.3f} ms"
    return f"{ms:.2f} ms"


# CSV_PATH     = r"E:\MSC_University Academics\CSE715 - NN & FS\Project\Dataset\NASA Bearing Dataset\combined.csv"
# MODEL_PATH   = r"E:\MSC_University Academics\CSE715 - NN & FS\Project\Dataset\NASA Bearing Dataset\vae_int8.tflite"
# THR_PATH     = r"E:\MSC_University Academics\CSE715 - NN & FS\Project\Dataset\NASA Bearing Dataset\deployment_thresholds.json"
# ROBUST_PATH  = r"E:\MSC_University Academics\CSE715 - NN & FS\Project\Dataset\NASA Bearing Dataset\robust_scaler.pkl"
CSV_PATH     = r"/home/jetson/desktop/CSE715/combined.csv"
MODEL_PATH   = r"/home/jetson/desktop/CSE715/vae_int8.tflite"
THR_PATH     = r"/home/jetson/desktop/CSE715/deployment_thresholds.json"
ROBUST_PATH  = r"/home/jetson/desktop/CSE715/robust_scaler.pkl"

# xAI knobs
TOP_K_PRINT          = 8
FD_DELTA_SCALED      = 0.20
CLIP_FD_TO_INT8_GRID = True
ERR_CONTRIB_MIN      = 20.0   

# 
STEP_DELAY_MS        = 500   
# ============================================================


def load_scaler_robust(robust_path: str):
    if not os.path.exists(robust_path):
        raise FileNotFoundError(
            f"Robust scaler not found at:\n  {robust_path}\n"
            "Re-export with use_robust=True to create it."
        )
    scaler = joblib.load(robust_path)

    def transform(X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X[None, :]
        return scaler.transform(X.astype(np.float32))

    def inverse_transform(Xs: np.ndarray) -> np.ndarray:
        if not isinstance(Xs, np.ndarray):
            Xs = np.asarray(Xs, dtype=np.float32)
        if Xs.ndim == 1:
            Xs = Xs[None, :]
        return scaler.inverse_transform(Xs.astype(np.float32))

    return scaler, transform, inverse_transform


def load_threshold_evt(thr_path: str) -> float:
    if not os.path.exists(thr_path):
        raise FileNotFoundError(f"Thresholds JSON not found: {thr_path}")
    with open(thr_path, "r") as f:
        d = json.load(f)
    return float(d.get("thr_evt"))


def qparams(detail):
    scale, zp = detail.get("quantization", (1.0, 0))
    if isinstance(scale, (list, tuple, np.ndarray)) and len(scale) > 0:
        scale = float(scale[0])
    return float(scale), int(zp)


def quantize_int8(x_float: np.ndarray, scale: float, zp: int) -> np.ndarray:
    q = np.round(x_float / scale + zp).astype(np.int32)
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize_int8(x_int8: np.ndarray, scale: float, zp: int) -> np.ndarray:
    return (x_int8.astype(np.float32) - float(zp)) * float(scale)


def run_model_mse(interpreter, in_det, out_det, in_scale, in_zp, out_scale, out_zp, x_scaled_1xF: np.ndarray):
    """Quantize -> invoke -> dequantize, return (y_scaled_1xF, mse_float)."""
    x_q = quantize_int8(x_scaled_1xF, in_scale, in_zp)
    interpreter.set_tensor(in_det["index"], x_q)
    interpreter.invoke()
    y_q = interpreter.get_tensor(out_det["index"])         
    y_scaled = dequantize_int8(y_q, out_scale, out_zp)     
    mse_val = float(np.mean((x_scaled_1xF - y_scaled) ** 2))
    return y_scaled, mse_val


# ---------- E
def explain_row_and_bearings(
    row_idx: int,
    x_row: np.ndarray,
    feat_names: list,
    transform,
    interpreter, in_det, out_det, in_scale, in_zp, out_scale, out_zp,
    thr_evt: float,
    fd_delta_scaled: float = 0.2,
    clip_to_int8: bool = True,
    top_k_print: int = 8,
    err_min: float = ERR_CONTRIB_MIN,
):
   
    x_scaled = transform(x_row[None, :])                   
    y_scaled, mse_val = run_model_mse(interpreter, in_det, out_det, in_scale, in_zp, out_scale, out_zp, x_scaled)
    errs = (x_scaled - y_scaled) ** 2                      


    top_err_idx = np.argsort(-e_vec)[:top_k_print]

    bearing_ids = set()
    for j in top_err_idx:
        if float(e_vec[j]) <= err_min:
            continue
        name = feat_names[j]
        m = re.match(r'^[Bb](\d+)[_\W]', name)
        if m:
            num = int(m.group(1))
            if 1 <= num <= 4:
                bearing_ids.add(num)

    decision = int(mse_val >= thr_evt)  # 1=Anomaly, 0=Normal
    return mse_val, decision, bearing_ids



GREEN = QtGui.QColor("#28a745")
RED   = QtGui.QColor("#dc3545")

class BearingItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, idx: int, center: QtCore.QPointF, radius: float = 60.0, healthy: bool = True):
        super().__init__(-radius, -radius, 2*radius, 2*radius)
        self.setPos(center)
        self.idx = idx
        self.setZValue(1)
        self.setPen(QtGui.QPen(QtCore.Qt.black, 2))
        self.setBrush(QtGui.QBrush(GREEN if healthy else RED))

        self.text = QtWidgets.QGraphicsSimpleTextItem(str(idx))
        font = QtGui.QFont(); font.setPointSize(25); font.setBold(True)
        self.text.setFont(font)
        br = self.text.boundingRect()
        self.text.setPos(center.x() - br.width()/2, center.y() - br.height()/2)
        self.text.setZValue(2)

    def set_healthy(self, healthy: bool):
        self.setBrush(QtGui.QBrush(GREEN if healthy else RED))


class BearingCanvas(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor("#f8f9fa")))
        self.scene = QtWidgets.QGraphicsScene(self); self.setScene(self.scene)
        self.scene.setSceneRect(0, 0, 900, 680)

        shaft = QtWidgets.QGraphicsRectItem(100, 300, 700, 80)
        shaft.setBrush(QtGui.QBrush(QtGui.QColor("#d1d5db")))
        shaft.setPen(QtGui.QPen(QtGui.QColor("#9ca3af"), 2))
        self.scene.addItem(shaft)

        positions = [QtCore.QPointF(180, 340), QtCore.QPointF(350, 340),
                     QtCore.QPointF(520, 340), QtCore.QPointF(690, 340)]
        self.bearings = {}
        for i, pos in enumerate(positions, start=1):
            b = BearingItem(i, pos, radius=45, healthy=True)
            self.scene.addItem(b); self.scene.addItem(b.text)
            self.bearings[i] = b

        self._add_legend()

    def _add_legend(self):
        title = QtWidgets.QGraphicsSimpleTextItem("Legend")
        f = QtGui.QFont(); f.setPointSize(14); f.setBold(True); title.setFont(f)
        green_box = QtWidgets.QGraphicsRectItem(0, 0, 18, 18); green_box.setBrush(GREEN)
        green_box.setPen(QtGui.QPen(QtCore.Qt.black, 1))
        green_txt = QtWidgets.QGraphicsSimpleTextItem("Normal")
        red_box = QtWidgets.QGraphicsRectItem(0, 0, 18, 18); red_box.setBrush(RED)
        red_box.setPen(QtGui.QPen(QtCore.Qt.black, 1))
        red_txt = QtWidgets.QGraphicsSimpleTextItem("Anomaly")
        green_txt.setFont(QtGui.QFont("Segoe UI", 11))
        red_txt.setFont(QtGui.QFont("Segoe UI", 11))
        title.setPos(780, 40); green_box.setPos(780, 70); green_txt.setPos(804, 68)
        red_box.setPos(780, 95); red_txt.setPos(804, 93)
        for item in [title, green_box, green_txt, red_box, red_txt]:
            self.scene.addItem(item)

    def set_bearing_state(self, bearing_id: int, healthy: bool):
        if bearing_id in self.bearings:
            self.bearings[bearing_id].set_healthy(healthy)

    def reset_all(self, healthy=True):
        for i in range(1, 5):
            self.set_bearing_state(i, healthy)


class ControlPanel(QtWidgets.QWidget):
    browseRequested = QtCore.pyqtSignal()
    startRequested = QtCore.pyqtSignal()
    stopRequested = QtCore.pyqtSignal()

    def __init__(self, thr_evt: float, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(480)

        layout = QtWidgets.QVBoxLayout(self)

        self.setStyleSheet("QPushButton { font-size: 14pt; }")

        layout.setContentsMargins(16, 28, 16, 16); layout.setSpacing(12)

        title = QtWidgets.QLabel("Dataset & Auto Prediction")
        title.setStyleSheet("font-size:22px; font-weight:600;")
        layout.addWidget(title)

   
        p_row = QtWidgets.QHBoxLayout()
        self.csv_edit = QtWidgets.QLineEdit()
        self.csv_edit.setPlaceholderText("Path to CSV with named features (+ GT)")
        self.csv_edit.setReadOnly(True)
        self.csv_btn = QtWidgets.QPushButton("Browse CSV…")
        self.csv_btn.clicked.connect(self.browseRequested.emit)
        p_row.addWidget(self.csv_edit); p_row.addWidget(self.csv_btn)
        layout.addLayout(p_row)

        self.row_info = QtWidgets.QLabel("Rows: N/A")
        layout.addWidget(self.row_info)

    
        btn_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start Auto Predict")
        self.stop_btn  = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.startRequested.emit)
        self.stop_btn.clicked.connect(self.stopRequested.emit)
        btn_row.addWidget(self.start_btn); btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        layout.addSpacing(12)
        sep = QtWidgets.QFrame(); sep.setFrameShape(QtWidgets.QFrame.HLine); sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(sep)

        thr_lbl = QtWidgets.QLabel(f"EVT Threshold (auto): <b>{thr_evt:.6f}</b>")
        layout.addWidget(thr_lbl)

  
        log_title = QtWidgets.QLabel("Log (latest prediction)")
        log_title.setStyleSheet("font-size:22px; font-weight:600;")
        layout.addWidget(log_title)
        self.log = QtWidgets.QTextEdit(); self.log.setFont(QtGui.QFont("Consolas", 14)); self.log.setReadOnly(True); self.log.setMinimumHeight(260)
        layout.addWidget(self.log)

        layout.addStretch(1)


    def set_csv_path(self, path: str):
        self.csv_edit.setText(path)

    def set_row_range(self, n_rows: int):
        self.row_info.setText(f"Rows: {n_rows} (valid indices: 0 … {max(0, n_rows-1)})")

    def clear_log(self):
        self.log.clear()

    def set_log(self, text: str):
        self.log.setPlainText(text)

    def set_running(self, running: bool):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.csv_btn.setEnabled(not running)


class RTState:
    """Holds model, scaler, CSV and utilities; supports CSV reload."""
    def __init__(self):
        self.csv_path = None
        self.df = None
        self.X = None
        self.gt = None   
        self.feat_names = []
        self.n_rows = 0
        self.n_feats = 0

   
        _, self.transform, self.inverse_transform = load_scaler_robust(ROBUST_PATH)
        self.thr_evt = load_threshold_evt(THR_PATH)

      
        self.interp = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interp.allocate_tensors()
        self.in_det  = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()[0]
        self.in_scale, self.in_zp   = qparams(self.in_det)
        self.out_scale, self.out_zp = qparams(self.out_det)

        if self.in_det["dtype"] != np.int8 or self.out_det["dtype"] != np.int8:
            print("[WARN] Model IO types are not int8. Script expects INT8<->INT8.")

    
        if os.path.exists(CSV_PATH):
            self.load_csv(CSV_PATH)

    def load_csv(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)

        self.gt = None
        feat_df = df.copy()
        if "GT" in feat_df.columns:
            self.gt = feat_df["GT"].values
            feat_df = feat_df.drop(columns=["GT"])

        X = feat_df.values.astype(np.float32)
        self.csv_path = path
        self.df = df
        self.X = X
        self.feat_names = list(feat_df.columns)
        self.n_rows, self.n_feats = X.shape

    def eval_row_and_bearings(self, row_idx: int):
        if self.X is None or self.n_rows == 0:
            return None, "CSV not loaded."
        if not (0 <= row_idx < self.n_rows):
            return None, f"Invalid row index {row_idx} (must be 0..{self.n_rows-1})."

        x0 = self.X[row_idx]
        t0_ns = time.perf_counter_ns()
        mse, decision, bearing_ids = explain_row_and_bearings(
            row_idx=row_idx,
            x_row=x0,
            feat_names=self.feat_names,
            transform=self.transform,
            interpreter=self.interp,
            in_det=self.in_det, out_det=self.out_det,
            in_scale=self.in_scale, in_zp=self.in_zp,
            out_scale=self.out_scale, out_zp=self.out_zp,
            thr_evt=self.thr_evt,
            fd_delta_scaled=FD_DELTA_SCALED,
            clip_to_int8=CLIP_FD_TO_INT8_GRID,
            top_k_print=TOP_K_PRINT,
            err_min=ERR_CONTRIB_MIN
        )
        elapsed_ms = (time.perf_counter_ns() - t0_ns) / 1_000_000.0
        gt_val = None if self.gt is None else self.gt[row_idx]
        return (mse, decision, bearing_ids, elapsed_ms, gt_val), None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, rt: RTState):
        super().__init__()
        self.rt = rt

        self.setWindowTitle("Bearing Anomaly Monitor — RT TFLite + XAI (Auto)")
        self.resize(1450, 720); self.setMinimumSize(1200, 680)

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        hbox = QtWidgets.QHBoxLayout(central); hbox.setContentsMargins(8, 8, 8, 8); hbox.setSpacing(8)

        self.canvas = BearingCanvas()
        self.controls = ControlPanel(thr_evt=self.rt.thr_evt)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.canvas); splitter.addWidget(self.controls)
        splitter.setStretchFactor(0, 3); splitter.setStretchFactor(1, 1)
        hbox.addWidget(splitter)


        self.controls.browseRequested.connect(self.on_browse_csv)
        self.controls.startRequested.connect(self.on_start_auto)
        self.controls.stopRequested.connect(self.on_stop_auto)

  
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(STEP_DELAY_MS)
        self.timer.timeout.connect(self._predict_next)

 
        self.cur_idx = 0
        self.running = False


        self.sum_ms = 0.0
        self.n_timed = 0

        self.correct = 0
        self.total   = 0
        self.tp = self.tn = self.fp = self.fn = 0

       
        if self.rt.csv_path:
            self.controls.set_csv_path(self.rt.csv_path)
            self.controls.set_row_range(self.rt.n_rows)

    # -------
    @QtCore.pyqtSlot()
    def on_browse_csv(self):
        if self.running:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            self.rt.load_csv(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", str(e))
            return

        # reset mare
        self.controls.set_csv_path(self.rt.csv_path)
        self.controls.set_row_range(self.rt.n_rows)
        self.canvas.reset_all(healthy=True)
        self.controls.clear_log()

    @QtCore.pyqtSlot()
    def on_start_auto(self):
        if self.running:
            return
        if self.rt.csv_path is None or self.rt.n_rows == 0:
            QtWidgets.QMessageBox.warning(self, "CSV not loaded", "Please load a CSV first.")
            return

        self.cur_idx = 0
        self.running = True
        self.controls.set_running(True)
        self.canvas.reset_all(healthy=True)
        self.controls.clear_log()

     
        self.sum_ms = 0.0
        self.n_timed = 0
        self.correct = 0
        self.total = 0
        self.tp = self.tn = self.fp = self.fn = 0

   
        self._predict_next()
        self.timer.start()

    @QtCore.pyqtSlot()
    def on_stop_auto(self):
        if not self.running:
            return
        self.timer.stop()
        self.running = False
        self.controls.set_running(False)

  
        if self.total > 0:
            acc_pct = 100.0 * self.correct / self.total
            summary = (
                f"Run finished — Accuracy: {self.correct}/{self.total} ({acc_pct:.2f}%)\n"
                f"TP={self.tp}, TN={self.tn}, FP={self.fp}, FN={self.fn}"
            )
            # A
            existing = self.controls.log.toPlainText()
            self.controls.set_log((existing + "\n" + summary).strip())
            QtWidgets.QMessageBox.information(self, "Run Summary", summary)


    def _predict_next(self):
        if not self.running:
            return
        if self.cur_idx >= self.rt.n_rows:
    
            self.on_stop_auto()
            return

        result, err = self.rt.eval_row_and_bearings(self.cur_idx)
        self.controls.clear_log()

        if err is not None:
            self.controls.set_log(err)
            self.on_stop_auto()
            return

        mse, decision, bearing_ids, elapsed_ms, gt_val = result


        self.sum_ms += elapsed_ms
        self.n_timed += 1
        avg_ms = self.sum_ms / self.n_timed if self.n_timed > 0 else elapsed_ms

   
        self.canvas.reset_all(healthy=True)
        for b in range(1, 5):
            self.canvas.set_bearing_state(b, healthy=(b not in bearing_ids))

   
        acc_line = "Accuracy: N/A (no GT)"
        match_line = "Match GT: N/A"
        gt_str = "N/A"
        if gt_val is not None:
            try:
                gt_int = int(gt_val)
            except Exception:
                gt_int = int(float(gt_val))
            pred_int = int(decision)  # 1=Anomaly, 0=Normal

            is_correct = (pred_int == gt_int)
            self.total += 1
            if is_correct:
                self.correct += 1

          
            if gt_int == 1 and pred_int == 1:
                self.tp += 1
            elif gt_int == 0 and pred_int == 0:
                self.tn += 1
            elif gt_int == 0 and pred_int == 1:
                self.fp += 1
            elif gt_int == 1 and pred_int == 0:
                self.fn += 1

            acc_pct = 100.0 * self.correct / self.total
            acc_line = f"Accuracy: {self.correct}/{self.total} ({acc_pct:.2f}%)"
            match_line = f"Match GT: {'Yes' if is_correct else 'No'}"
            gt_str = f"{gt_int} ({'Anomaly' if gt_int==1 else 'Normal'})"

        log_text = (
            f"Row {self.cur_idx}\n"
            f"MSE: {mse:.6f}\n"
            # f"Threshold: {self.rt.thr_evt:.6f}\n"
            f"Decision: {'Anomaly' if decision==1 else 'Normal'} ({decision})\n"
            f"Ground Truth: {gt_str}\n"
            # f"{match_line}\n"
            f"{acc_line}\n"
            f"Time: {fmt_time_ms(elapsed_ms)}\n"
            f"Time Avg: {fmt_time_ms(avg_ms)})\n"
            f"Bearings flagged: {'Yes' if len(bearing_ids) > 0 else 'No'}"
        )
        self.controls.set_log(log_text)

        self.cur_idx += 1
        

def main():
    app = QtWidgets.QApplication(sys.argv)
    rt = RTState()
    win = MainWindow(rt)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
